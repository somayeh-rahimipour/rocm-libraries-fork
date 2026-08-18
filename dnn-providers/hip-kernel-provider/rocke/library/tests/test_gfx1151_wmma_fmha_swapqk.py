# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1151 transposed-QK WMMA FMHA-forward (swapqk) tests -- CPU only.

Covers the surface that does not need a gfx1151 device:
  * the shipped ``SwapQKCfg`` defaults ARE the measured-winning configuration
    (the regression guard that matters most: the kernel's whole value is that
    one config, and a default silently drifting off it is invisible at runtime);
  * build + lower for the production shape and the levers that change codegen
    structure (row-major V, causal, D64, persistent grid);
  * ``is_valid_spec`` rejects the shapes the kernel cannot emit;
  * the launch-geometry helpers and the V-relay layout transform;
  * the register budget, asserted from a real code-object compile.

The kernel COMPILES on any host (comgr targets gfx1151 regardless of the build
GPU); only execution needs the board, so nothing here is GPU-gated. The compile
and disassembly cases skip when the toolchain is unavailable.

    python3 -m pytest library/tests/test_gfx1151_wmma_fmha_swapqk.py
"""

from __future__ import annotations

import unittest

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    is_valid_spec,
    swapqk_causal_kv_stop,
    swapqk_grid,
    swapqk_num_work_items,
    swapqk_transpose_v,
)

# The production shape the kernel was tuned and hardware-validated on.
_D = 128
_HQ = 24


def _prod_cfg(**kw) -> SwapQKCfg:
    kw.setdefault("head_size", _D)
    kw.setdefault("num_query_heads", _HQ)
    return SwapQKCfg(**kw)


class TestSwapQKDefaults(unittest.TestCase):
    """The defaults are the product. Pin them."""

    def test_defaults_are_the_measured_winner(self):
        # This kernel ships exactly one configuration; every knob below was
        # chosen by hardware A/B (see the field docs). A default drifting off
        # this set still builds and still computes correct attention, so it
        # would pass every numeric test while quietly costing throughput --
        # which is precisely how an earlier production wrapper ended up
        # building block_n=32 with v_transposed and qk_douter off.
        cfg = _prod_cfg()
        self.assertEqual(cfg.n_waves, 2)
        self.assertEqual(cfg.block_n, 64)
        self.assertEqual(cfg.qk_ilp, 2)
        self.assertEqual(cfg.q_block, 1)
        self.assertEqual(cfg.sched_mode, "pingpong")
        self.assertTrue(cfg.buffer_gather)
        self.assertTrue(cfg.dual_gather)
        self.assertTrue(cfg.lazy_rescale)
        self.assertTrue(cfg.fast_exp2)
        self.assertTrue(cfg.v_transposed)
        # qk_douter's documented +3.3% did not reproduce and its sign was wrong:
        # -4.7% at H24 MHA dense and -35.7% at Hq32/Hk8 GQA causal, both
        # confirmed in SQ_BUSY_CYCLES at an identical SQ_WAVES.
        self.assertFalse(cfg.qk_douter)
        # gqa_fuse is opt-in. It is a large win where it applies but it widens
        # the CTA, so callers choose it rather than inheriting it silently.
        self.assertEqual(cfg.gqa_fuse, 1)

    def test_experimental_levers_default_off(self):
        # Everything with a recorded regression stays off, so the default build
        # is the winner rather than the newest lever someone was measuring.
        cfg = _prod_cfg()
        for knob in (
            "pipeline",
            "q_hoist",
            "q_lds",
            "kv_lds",
            "o_f16",
            "d16hi",
            "o_nt",
            "q_nt",
            "static_shape",
            "prefetch_v",
            "k_dual",
        ):
            self.assertFalse(getattr(cfg, knob), f"{knob} must default off")
        self.assertEqual(cfg.v_kblock, 0)
        self.assertEqual(cfg.v_prefetch, 0)
        self.assertEqual(cfg.bcast_group, 0)
        self.assertEqual(cfg.num_persistent, 0)
        self.assertEqual(cfg.iglp, -1)
        self.assertIsNone(cfg.waves_per_eu)

    def test_default_cfg_is_valid(self):
        ok, why = is_valid_spec(_prod_cfg())
        self.assertTrue(ok, why)

    def test_kernel_name_encodes_the_winning_knobs(self):
        # The name is how a hsaco on the board is traced back to its config, so
        # the winning tokens have to survive into it.
        name = _prod_cfg().kernel_name()
        for token in ("H128", "HQ24", "w2", "pingpong", "ilp2", "bn64"):
            self.assertIn(token, name)
        for token in ("vt", "dual", "buf", "lazy", "fexp", "qkno"):
            self.assertIn(f"_{token}", name)
        # F==1 names stay byte-identical to the pre-fusion kernel.
        self.assertNotIn("gf", name)

    def test_fused_and_unfused_names_cannot_collide(self):
        # The artifact cache is keyed on this string. If a fused build shared a
        # key with the unfused one the cache would serve the wrong binary and
        # every measurement after that point would be a lie -- and it would
        # still PASS numerically, because both kernels compute correct
        # attention. Nothing else in the suite catches that.
        base = _prod_cfg(num_kv_heads=6).kernel_name()
        names = {base}
        for f in (2, 4):
            n = _prod_cfg(num_kv_heads=6, gqa_fuse=f).kernel_name()
            self.assertIn(f"gf{f}", n)
            self.assertNotIn(n, names)
            names.add(n)
        self.assertEqual(len(names), 3)


class TestSwapQKValidity(unittest.TestCase):
    def test_rejects_non_gfx1151_arch(self):
        ok, why = is_valid_spec(_prod_cfg(), arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("gfx1151", why)

    def test_rejects_head_size_not_multiple_of_32_under_dual_gather(self):
        # dual_gather pairs adjacent d-subtiles, so n_dk must be even.
        ok, why = is_valid_spec(_prod_cfg(head_size=48))
        self.assertFalse(ok)
        self.assertIn("head_size", why)

    def test_rejects_block_n_below_the_buffer_gather_floor(self):
        # The backend stops batching the buffer gather below block_n=32.
        ok, why = is_valid_spec(_prod_cfg(block_n=16))
        self.assertFalse(ok)
        self.assertIn("block_n", why)

    def test_rejects_block_n_off_the_16_grid(self):
        ok, _ = is_valid_spec(_prod_cfg(block_n=48))
        self.assertTrue(ok)  # 48 is a legal multiple of 16 at/above the floor
        ok, why = is_valid_spec(_prod_cfg(block_n=40))
        self.assertFalse(ok)
        self.assertIn("block_n", why)

    def test_rejects_unsupported_wave_count_and_mask(self):
        ok, why = is_valid_spec(_prod_cfg(n_waves=4))
        self.assertFalse(ok)
        self.assertIn("n_waves", why)
        ok, why = is_valid_spec(_prod_cfg(mask_mode="sliding"))
        self.assertFalse(ok)
        self.assertIn("mask_mode", why)

    def test_accepts_gqa_fuse_that_divides_the_ratio(self):
        # HQ24 / HK6 -> ratio 4, so F in {1,2,4}.
        for f in (1, 2, 4):
            ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=f))
            self.assertTrue(ok, f"gqa_fuse={f}: {why}")

    def test_rejects_gqa_fuse_that_does_not_divide_the_ratio(self):
        # A stale F would split a CTA's waves across two KV heads while the
        # kv_head decode still reads block_id_y -- silently wrong attention.
        for f in (3, 8):
            ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=f))
            self.assertFalse(ok, f"gqa_fuse={f} must be rejected")
            self.assertIn("gqa_fuse", why)

    def test_rejects_gqa_fuse_without_kv_heads(self):
        # Under MHA there is nothing to fuse; the ratio is 1.
        ok, why = is_valid_spec(_prod_cfg(gqa_fuse=2))
        self.assertFalse(ok)
        self.assertIn("num_kv_heads", why)

    def test_rejects_gqa_fuse_with_persistent_or_q_lds(self):
        # Both decode work-items or LDS slabs from wave_id / a one-head-per-CTA
        # work count, neither of which survives fusion.
        ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=4, num_persistent=960))
        self.assertFalse(ok)
        self.assertIn("num_persistent", why)
        ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=4, q_lds=True))
        self.assertFalse(ok)
        self.assertIn("q_lds", why)

    def test_builder_raises_on_an_invalid_config(self):
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(_prod_cfg(block_n=24))

    def test_builder_rejects_incompatible_knob_pairs(self):
        # v_transposed rides the buffer-descriptor gather; without it there is
        # no path to emit, so this must fail loudly rather than mis-address V.
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(_prod_cfg(v_transposed=True, buffer_gather=False))


class TestSwapQKLowering(unittest.TestCase):
    def _lower(self, cfg):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        return lower_kernel_to_llvm(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
        )

    def test_production_config_lowers_to_the_rdna35_wmma_intrinsic(self):
        # gfx1151 has exactly one matrix atom; a K=32 intrinsic here would mean
        # the atom selection silently picked a gfx12 form.
        ll = self._lower(_prod_cfg())
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)
        self.assertNotIn("16x16x32", ll)

    def test_production_config_uses_the_buffer_descriptor_d16_gather(self):
        # The buffer gather is worth a double-digit percentage and is fragile:
        # it only wins at w2/bn>=32. Assert the emitted form is the buffer one,
        # not the flat fallback, so a lowering change cannot quietly demote it.
        ll = self._lower(_prod_cfg())
        self.assertIn("raw.ptr.buffer.load", ll)

    def test_lane_broadcast_is_emitted_for_the_dual_subtile_gather(self):
        ll = self._lower(_prod_cfg())
        self.assertIn("permlanex16", ll)

    def test_row_major_v_also_lowers(self):
        # The documented escape hatch for callers that cannot pre-transpose V.
        ll = self._lower(_prod_cfg(v_transposed=False))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_causal_and_gqa_lower(self):
        ll = self._lower(_prod_cfg(mask_mode="causal", num_kv_heads=4))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_d64_lowers(self):
        ll = self._lower(SwapQKCfg(head_size=64, num_query_heads=8))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_persistent_grid_lowers_and_needs_seqlen_at_build_time(self):
        # num_persistent bakes the work-item count in, so the builder must
        # refuse to guess it.
        cfg = _prod_cfg(num_persistent=960)
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(cfg, arch="gfx1151")
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        ll = lower_kernel_to_llvm(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151", seqlen_q=2048, batch=1),
            arch="gfx1151",
        )
        self.assertIn("atomicrmw", ll)

    def test_gqa_fuse_lowers_and_keeps_the_head_decode_scalar(self):
        # workitem.id.x is a divergence source, so LLVM cannot prove tid/32 is
        # wave-uniform; without the forced readfirstlane the head and query-row
        # addressing both go vector and the win evaporates. Assert the pin
        # survived into the IR rather than trusting the builder call.
        base = _prod_cfg(num_kv_heads=6, mask_mode="causal")
        ll = self._lower(_prod_cfg(num_kv_heads=6, gqa_fuse=4, mask_mode="causal"))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)
        self.assertGreater(
            ll.count("readfirstlane"), self._lower(base).count("readfirstlane")
        )

    def test_distinct_configs_build_distinct_kernels(self):
        a = build_wmma_fmha_swapqk(_prod_cfg(), arch="gfx1151")
        b = build_wmma_fmha_swapqk(_prod_cfg(block_n=32), arch="gfx1151")
        self.assertNotEqual(a.name, b.name)


class TestSwapQKLaunchGeometry(unittest.TestCase):
    def test_grid_is_q_blocks_by_heads_by_batch(self):
        cfg = _prod_cfg()
        # w2 q_block1 -> 32 query rows per CTA.
        self.assertEqual(swapqk_grid(cfg, seqlen_q=16384, batch=1), (512, _HQ, 1))
        self.assertEqual(swapqk_grid(cfg, seqlen_q=2048, batch=3), (64, _HQ, 3))

    def test_single_wave_halves_the_rows_per_cta(self):
        self.assertEqual(
            swapqk_grid(_prod_cfg(n_waves=1), seqlen_q=1024, batch=1), (64, _HQ, 1)
        )

    def test_work_item_count_matches_the_one_shot_grid(self):
        cfg = _prod_cfg()
        qb, h, b = swapqk_grid(cfg, seqlen_q=4096, batch=2)
        self.assertEqual(swapqk_num_work_items(cfg, seqlen_q=4096, batch=2), qb * h * b)

    def test_block_size_tracks_the_wave_count(self):
        self.assertEqual(_prod_cfg().block_size, 64)
        self.assertEqual(_prod_cfg(n_waves=1).block_size, 32)

    def test_kv_heads_defaults_to_mha(self):
        self.assertEqual(_prod_cfg().kv_heads, _HQ)
        self.assertEqual(_prod_cfg(num_kv_heads=4).kv_heads, 4)

    def test_gqa_fuse_widens_the_cta_without_widening_the_query_rows(self):
        # This is the whole design: the extra waves serve extra HEADS, so each
        # wave still owns exactly one head's 128-VGPR O accumulator and the
        # register budget is untouched. If q_rows_per_cta ever tracked gqa_fuse
        # the kernel would spill and the win would invert.
        base = _prod_cfg(num_kv_heads=6)
        for f in (1, 2, 4):
            cfg = _prod_cfg(num_kv_heads=6, gqa_fuse=f)
            self.assertEqual(cfg.block_size, base.block_size * f)
            self.assertEqual(cfg.q_rows_per_cta, base.q_rows_per_cta)

    def test_gqa_fuse_folds_heads_into_the_y_axis(self):
        # Total waves launched must be invariant -- that equality is the control
        # for every fusion A/B: an unmatched SQ_WAVES means the two arms did
        # different amounts of work and the comparison means nothing.
        base = _prod_cfg(num_kv_heads=6)
        qb, h, bz = swapqk_grid(base, seqlen_q=2048, batch=2)
        for f in (2, 4):
            cfg = _prod_cfg(num_kv_heads=6, gqa_fuse=f)
            g = swapqk_grid(cfg, seqlen_q=2048, batch=2)
            self.assertEqual(g, (qb, h // f, bz))
            self.assertEqual(
                g[0] * g[1] * g[2] * cfg.block_size,
                qb * h * bz * base.block_size,
            )


class TestSwapQKCausalTrim(unittest.TestCase):
    """The kv-block trim is the one place a wrong answer is silent.

    Over-inclusion costs a block of wasted work; the inline mask still zeroes
    it. Under-inclusion drops real attention weight from the LAST query rows of
    a CTA, which no shape check and no compile catches -- only a numeric
    comparison would, and only at the exact (n_waves, q_block, block_n) triple
    that trips it.
    """

    def _cover(self, cfg, q_group: int) -> bool:
        """Does the trim reach the diagonal of this CTA's last query row?"""
        last_row = q_group * cfg.q_rows_per_cta + cfg.q_rows_per_cta - 1
        return swapqk_causal_kv_stop(cfg, q_group) > last_row // cfg.block_n

    def test_trim_covers_every_query_row_across_the_knob_grid(self):
        for n_waves in (1, 2):
            for q_block in (1, 2):
                for block_n in (16, 32, 48, 64):
                    cfg = _prod_cfg(
                        mask_mode="causal",
                        n_waves=n_waves,
                        q_block=q_block,
                        block_n=block_n,
                    )
                    for g in range(8):
                        self.assertTrue(
                            self._cover(cfg, g),
                            f"w{n_waves} qb{q_block} bn{block_n} g{g} drops a kv block",
                        )

    def test_trim_regression_at_the_config_the_old_form_broke(self):
        # The pre-fix form used the wave span 16*n_waves and omitted the q_block
        # factor. At w2/qb2/bn16 that returns 4g+3 where 4g+4 is required.
        cfg = _prod_cfg(mask_mode="causal", n_waves=2, q_block=2, block_n=16)
        self.assertEqual(cfg.q_rows_per_cta, 64)
        for g in range(4):
            self.assertEqual(swapqk_causal_kv_stop(cfg, g), 4 * g + 4)

    def test_trim_is_tight_not_merely_safe(self):
        # A correct-but-loose trim would pass the coverage test above while
        # quietly paying for kv blocks that are entirely above the diagonal.
        for block_n in (16, 32, 64):
            cfg = _prod_cfg(mask_mode="causal", block_n=block_n)
            for g in range(6):
                last_row = g * cfg.q_rows_per_cta + cfg.q_rows_per_cta - 1
                self.assertEqual(
                    swapqk_causal_kv_stop(cfg, g), last_row // block_n + 1
                )


class TestSwapQKAdaptiveBlockN(unittest.TestCase):
    """The torch op picks block_n from the key length.

    The divisibility half of this is a CORRECTNESS gate, not a perf knob: the kv
    loop bound is ``seqlen_k // block_n``, so a non-divisible launch drops the
    tail silently. The threshold half encodes the measured L2 crossover.
    """

    @staticmethod
    def _pick():
        from rocke.instances.gfx1151.wmma_fmha_swapqk import _pick_block_n

        return _pick_block_n

    def test_table_thresholds(self):
        pick = self._pick()
        self.assertEqual(pick(2048), 64)
        self.assertEqual(pick(3072), 64)
        self.assertEqual(pick(4096), 128)
        self.assertEqual(pick(8192), 128)

    def test_demotes_when_the_wider_tile_does_not_divide(self):
        # Past the threshold but not 128-aligned: must fall back rather than
        # truncate 64 keys.
        pick = self._pick()
        self.assertEqual(pick(4160), 64)

    def test_rejects_lengths_no_tile_divides(self):
        # 2080 is 32-aligned, which the old vLLM gate accepted while running
        # bn64 -- that silently dropped the last 32 keys.
        pick = self._pick()
        self.assertEqual(pick(2080), 0)
        self.assertEqual(pick(48), 0)

    def test_env_pin_still_honours_divisibility(self):
        import rocke.instances.gfx1151.wmma_fmha_swapqk as inst

        prev = inst._BLOCK_N
        try:
            inst._BLOCK_N = "128"
            self.assertEqual(inst._pick_block_n(8192), 128)
            # A pin overrides the length thresholds but NOT divisibility: 576 is
            # 64-aligned, which "auto" would happily serve, and the pin must not.
            self.assertEqual(inst._pick_block_n(2048), 128)
            self.assertEqual(inst._pick_block_n(576), 0)
        finally:
            inst._BLOCK_N = prev

    def test_the_two_tiles_cannot_share_a_cache_key(self):
        # An artifact cache collision here would serve bn64 code for a bn128
        # config, making every measurement downstream a lie.
        self.assertNotEqual(
            _prod_cfg(block_n=64).kernel_name(),
            _prod_cfg(block_n=128).kernel_name(),
        )


class TestSwapQKVRelay(unittest.TestCase):
    """swapqk_transpose_v is the caller's contract for the default layout."""

    def test_transpose_maps_bshd_to_bhds_elementwise(self):
        np = __import__("numpy")
        v = np.arange(2 * 6 * 3 * 4, dtype=np.float16).reshape(2, 6, 3, 4)
        vt = swapqk_transpose_v(v)
        self.assertEqual(vt.shape, (2, 3, 4, 6))  # [B,S,H,D] -> [B,H,D,S]
        self.assertTrue(vt.flags["C_CONTIGUOUS"])
        for b, s, h, d in ((0, 0, 0, 0), (1, 5, 2, 3), (0, 3, 1, 2)):
            self.assertEqual(vt[b, h, d, s], v[b, s, h, d])

    def test_key_blocked_relay_shape_and_contents(self):
        np = __import__("numpy")
        kb = 2
        v = np.arange(1 * 4 * 2 * 3, dtype=np.float16).reshape(1, 4, 2, 3)
        vt = swapqk_transpose_v(v, kblock=kb)
        self.assertEqual(vt.shape, (1, 2, 4 // kb, 3, kb))  # [B,H,S/KB,D,KB]
        for s in range(4):
            self.assertEqual(vt[0, 1, s // kb, 2, s % kb], v[0, s, 1, 2])

    def test_key_blocked_relay_rejects_a_ragged_seqlen(self):
        np = __import__("numpy")
        v = np.zeros((1, 5, 2, 3), dtype=np.float16)
        with self.assertRaises(ValueError):
            swapqk_transpose_v(v, kblock=2)


class TestSwapQKCodeObject(unittest.TestCase):
    """Compile to a real code object; no GPU needed, comgr cross-targets."""

    def _compile(self, cfg):
        try:
            from rocke.helpers.compile import compile_kernel
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"compile toolchain unavailable: {e}")
        try:
            return compile_kernel(
                build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
            )
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"gfx1151 comgr compile unavailable: {e}")

    def test_production_config_compiles(self):
        self.assertGreater(self._compile(_prod_cfg()).hsaco_bytes, 0)

    def test_register_budget_holds(self):
        # The shipped config sits at 199 VGPR in a 208 granule = 7 waves/SIMD,
        # with nothing spilled. Crossing 208 costs a wave; spilling at all has
        # measured worse than the wave is worth. Both are invisible without
        # this assertion, so pin the budget rather than the exact count.
        art = self._compile(_prod_cfg())
        res = _resources(self, art)
        self.assertIsNotNone(res.vgpr_count, "no VGPR count in the code object")
        self.assertLessEqual(res.vgpr_count, 208, "lost a wave: VGPR past 7/SIMD")
        self.assertEqual(res.scratch_bytes or 0, 0, "spilled to scratch")

    def test_d64_has_register_headroom(self):
        # D64 halves the O accumulator, which is what makes the otherwise
        # register-blocked levers (pipeline, q_block=2) fit there and not here.
        res = _resources(
            self, self._compile(SwapQKCfg(head_size=64, num_query_heads=8))
        )
        self.assertIsNotNone(res.vgpr_count)
        self.assertLess(res.vgpr_count, 192)
        self.assertEqual(res.scratch_bytes or 0, 0)


def _resources(case: unittest.TestCase, art):
    """Decode the code object's resource note, skipping if tools are missing."""
    import os
    import tempfile

    try:
        from rocke.analysis.isa import analyze_hsaco
    except Exception as e:  # pragma: no cover - env-dependent
        case.skipTest(f"isa analysis unavailable: {e}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.hsaco")
        with open(path, "wb") as f:
            f.write(art.hsaco)
        try:
            return analyze_hsaco(path).resources
        except FileNotFoundError as e:  # pragma: no cover - env-dependent
            case.skipTest(f"disassembler unavailable: {e}")


if __name__ == "__main__":
    unittest.main()
