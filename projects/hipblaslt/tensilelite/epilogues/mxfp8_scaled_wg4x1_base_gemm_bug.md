# MX-Scaled FP8 Subtile GEMM — MIWaveGroup[0]=4 (WG4x1) Base-GEMM Correctness Bug

**Date:** 2026-08-19
**GPU:** gfx950
**Status:** Worked around (Option A guard in place); proper fix (Option B) pending.

## Summary

The base MX-scaled fp8 subtile GEMM produces **incorrect accumulators** when
`MIWaveGroup[0] > 2` (confirmed for `MIWaveGroup[0]=4`, i.e. the WG4x1 wave
group). This is a defect in the base GEMM, **independent of the PartialRMS
fused epilogue**. The 14 remaining `gemm_partial_rms_scaled_mxfp8_k1.yaml`
failures (the `MIWT4_16 / WG64_4_1` solution, all sizes, PGR=0 and PGR=1) are
downstream symptoms: PartialRMS consumes the already-wrong accumulator, so both
the `[d]` output and the `[partialBuf]` reduction come out wrong.

## Proof it is the base GEMM, not PartialRMS

Running the exact failing tile `[16,16,128,1,1,4,16,4,1]`
(MIWaveTile[4,16] / MIWaveGroup[4,1]) as a plain MX-scaled fp8 GEMM with
**no PartialRMS** fails on the `[d]` output tensor, while the
`[16,16,128,1,1,8,8,2,2]` (MIWaveGroup[2,2]) control passes every size:

| tile | 256x256 / K256 PGR0 | 256x256 / K256 PGR1 | 512x512 / K512 | 1024x1024 / K512 |
|------|------|------|------|------|
| MIWaveTile[4,16] MIWaveGroup[4,1] | PASS | FAIL | FAIL (both PGR) | FAIL (both PGR) |
| MIWaveTile[8,8] MIWaveGroup[2,2] | PASS | PASS | PASS | PASS |

## Failure pattern

- The discriminator is `MIWaveGroup[0]` (wg_m): wg_m in {1,2} pass, wg_m=4 fails.
  WG1x4 (MIWaveGroup[1,4], same MT256x256, equally asymmetric) passes; WG2x2
  passes even though it runs the same wg_m>1 cross-wave epilogue path.
- The error rate scales with K (body-iteration count): small K / PGR=0 may pass,
  larger K fails ~20-50%.
- **Both** the PGR=1 prefetch path and the PGR=0 single-partition path are
  broken. Because PGR=0 forces `numPartitions == 1`
  (`LogicalScheduler.py`, the `pgr == 0` assert), this is beyond the previously
  documented wrap-LR prefetch race — it reaches the core MX-scale local-read /
  codegen path for the mma_m=4 (MIWaveTile[0]=4) tile shape.
- Note: `PrefetchGlobalRead=2` is not a workaround — it is rejected for this
  DepthU=256 / dataDU=128 multi-DU config.

## Already ruled out

- **PartialRMS emitter (`SubtilePartialRMSEmit.py`) is correct.** The failing
  kernel was disassembled and verified instruction-by-instruction: the
  square-sum reads all 256 accumulator AGPRs with the correct
  `acc_idx = (n*mma_m+m)*rows_per_lane+k` layout; the XOR butterfly and the
  wg_m=4 cross-wave LDS reduction (strideW=4096, laneSlotBytes=64, 4 read
  iterations, `readBaseWave = waveId XOR waveM`) are correct and match the
  store's row/column mapping.
- **The LDS-reservation fix is correct and necessary.** `getLdsSize` returns
  `LdsNumBytes`, so the reservation bump genuinely sizes the group segment; it
  fixed the pre-fix ~90% overflow errors for wg_m>1 / wg_n=1.

## Files involved (where the real fix lives)

- `Tensile/Components/Subtile/LogicalScheduler.py` — subtile K-loop scheduling:
  partition sizing (`_normalize_partition_sizes`, `numPartitionsM/N`),
  local-read placement, and the multi-DU wait-count / free-list logic.
- `Tensile/Components/Subtile/SubtileScaleEmit.py` — MX scale VGPR/LDS layout
  and application into the MMA.
- `Tensile/Components/Subtile/SubtileLREmit.py` — local-read addressing for the
  A/B tiles and scales.

## Reproduction

Base GEMM (no PartialRMS) isolation repro:

```bash
cd /home/fmoracor/rocm-libraries/projects/hipblaslt
source ~/.tensile/bin/activate
LD_LIBRARY_PATH=/opt/rocm/lib PYTHONPATH=tensilelite \
python tensilelite/Tensile/bin/Tensile \
  /tmp/base_wg4x1_nopartialrms.yaml /tmp/base_wg4x1 \
  > /tmp/base_wg4x1.log 2>&1
grep -E ',(PASSED|FAILED),' /tmp/base_wg4x1.log
```

The minimal repro YAML (recreate at `/tmp/base_wg4x1_nopartialrms.yaml` if gone)
is a plain MX-scaled fp8 TN GEMM (`DataType: F8`, `DestDataType: b`,
`ComputeDataType: s`, `MXBlockA: 32`, `MXBlockB: 32`, `TransposeA: True`,
`UseSubtileImpl: True`, `StreamK: 3`, `StreamKForceDPOnly: 1`, `DepthU: 256`,
`ScheduleIterAlg: 3`, `PrefetchGlobalRead: [0,1]`) forking two
MatrixInstructions:

```
- [16, 16, 128, 1, 1,  4, 16, 4, 1]  # MT256x256, MIWaveTile[4,16], MIWaveGroup[4,1]  (FAILS)
- [16, 16, 128, 1, 1,  8,  8, 2, 2]  # MT256x256, MIWaveTile[8,8],  MIWaveGroup[2,2]  (control, PASSES)
```

over sizes `[256,256,1,256]`, `[512,512,1,512]`, `[1024,1024,1,512]`.

The PartialRMS-level repro is
`tensilelite/Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_scaled_mxfp8_k1.yaml`.

## Workaround in place (Option A)

`Tensile/SolutionStructs/Solution.py` — `_validateSubtileMXWaveGroup` rejects
any MX-scaled (`MXBlockA` or `MXBlockB`) subtile solution on gfx950 with
`MIWaveGroup[0] > 2`. Called from the general subtile-validation site next to
`_validateSubtileGRKPartition`. This keeps the offending kernel out of the
generated logic while leaving the YAMLs comprehensive (they still record the
broken tile). The `[16,16,128,1,1,4,16,4,1]` line remains in the PartialRMS
YAMLs on purpose.

## Where to start (Option B)

1. Reproduce with the base (no-PartialRMS) YAML above — it is far faster to
   iterate than the full PartialRMS suite and removes the epilogue as a
   variable.
2. Start with **PGR=0** on `[512,512,1,512]` (fails with `numPartitions == 1`,
   so prefetch/partition scheduling is out of the picture — the bug is in the
   single-partition MX-scale local-read/codegen path for mma_m=4).
3. In `LogicalScheduler.py`, focus on how the MX-scale local reads and MMA slots
   are laid out when `numMFMATilesM` (mma_m) = 4 and `numMFMATilesN` (mma_n) =
   16 (contrast the passing MIWaveGroup[2,2] shape mma_m=8/mma_n=8). Cross-check
   the scale-tile-to-MMA-tile mapping in `SubtileScaleEmit.py` /
   `SubtileLREmit.py` for the same shape.
4. Then re-enable prefetch (PGR=1) and confirm the wrap-LR / multi-DU
   wait-count logic also holds for wg_m=4.
5. Once fixed, remove `_validateSubtileMXWaveGroup` (or relax the `> 2` bound)
   and regenerate the `partialrms_scaled_mxfp8_k1` logic from all solutions.
