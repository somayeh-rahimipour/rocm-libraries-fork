# rocKE SwapQK prefill attention on gfx1151 (Strix Halo)

Reducing Qwen3-8B prefill latency on the Strix Halo iGPU by serving attention with
rocKE's `wmma_fmha_swapqk` kernel in place of vLLM's Triton Flash-Attention, with every
improvement validated end-to-end in a production vLLM v1 serving path rather than in
isolated microbenchmarks alone.

---

## Headline

**The shipped configuration is `qk_douter=False` + `gqa_fuse=4` + `k_lds` + `v_paged`,
at `block_n=64`.**

| | at the start | shipped today |
|---|---|---|
| **Isolated kernel**, S=8192 causal, Hq32/Hk8/D128 | 163.2 ms — **6.1× slower than Triton** | 20.9 ms — **1.14× faster than Triton** |
| **Isolated kernel**, S=2048 | 3.92 ms — 2.1× slower than Triton | 1.11 ms — **1.61× faster than Triton** |
| **End-to-end**, Qwen3-8B fp16 prefill TTFT | **−7.7% regression** vs a matched Triton control | **+3.5% faster than Triton** at S=8192 |

- **Kernel: ~7.8× faster** than the config that shipped when this work started
  (163.2 ms → 20.9 ms at S=8192), and it now beats Triton at **every** measured length
  (1.57× / 1.61× / 1.34× / 1.14× at S = 1024 / 2048 / 4096 / 8192).
- **End-to-end: ~3.5% off TTFT** at an 8192-token prompt (7022 → 6775 ms graph mode,
  7066 → 6817 ms eager), and never slower than Triton at any length. The e2e number is
  much smaller than the kernel number because **attention is only ~5–15% of prefill on
  this part** — see [§4](#4-end-to-end-results).

> The 7.8× is a *composed* figure spanning two measurement sessions. Each individual
> step below is measured within one interleaved session; the sessions overlap on a
> common config (`qk_douter=False, gqa_fuse=4, bn64`: 41.0 ms vs 37.7 ms) and agree to
> within the ~9–13% inter-session clock drift this box exhibits. Never quote a number
> stitched across sessions as if it were a single measurement.

---

## 1. The setup

**Hardware.** AMD Strix Halo, gfx1151, RDNA3.5, 40 CUs, wave32, 103 GiB unified LPDDR5X
at ~102 GB/s, 2 MB L2, 32 KB L0 per CU, 64 KB LDS per CU. ROCm 7.2.1,
torch 2.11.0+rocm7.2.

**Model.** Qwen3-8B fp16, 36 decoder layers, GQA with 32 query heads / 8 KV heads,
head dim 128, causal.

**Baselines.** vLLM's `_fwd_kernel` from `ops/triton_prefill_attention.py` (dense
Triton FA), and — for e2e — the whole vLLM prefill path with attention routed to it.

**The kernel.** `wmma_fmha_swapqk` computes `Sᵀ = K × Qᵀ` instead of `S = Q × Kᵀ`. That
puts the query index on `lane % 16`, so the C→A transpose of `P` before the `P×V` GEMM
becomes **register-local** (`permlanex16` + 2× `v_perm_b32`) — no LDS, no barrier. On
`v_wmma_f32_16x16x16_f16` that is a real structural advantage: 1019 inner-loop
instructions at 72% VALU density and zero barriers, against Triton's 3218 at 34% with
4 barriers and 1168 LDS ops per iteration.

---

## 2. How rocKE was integrated with vLLM

vLLM's backend auto-selection on this box resolves to `ROCM_ATTN`
(`RocmAttentionBackend` / `RocmAttentionImpl`). The rocKE backend is a **subclass of
that exact impl**, dropped in as
`vllm/v1/attention/backends/rocm_rocke_attn.py` and selected with
`VLLM_ATTENTION_BACKEND=ROCKE_GFX1151`. A vendored copy lives at
[`rocke/integrations/vllm/rocm_rocke_attn.py`](../../../../integrations/vllm/rocm_rocke_attn.py).

Because it subclasses the impl vLLM would have chosen anyway, the metadata builder, the
KV-cache shape and the KV-cache write path (`forward_includes_kv_cache_update = False`;
`do_kv_cache_update` is called by `attention.py`, not by `forward`) are all inherited
unchanged. **The only thing that differs is which prefill kernel runs.** That is what
makes the A/B honest.

**Dispatch.** `forward` checks eligibility, then per request:

1. `_full_prefill_layout` — every request must be a full prefill with zero cached
   context (`query_start_loc` diffs equal `seq_lens`).
2. `kv_cache_dtype in ("auto", "float16")`, `alibi_slopes is None`,
   `sliding_window == (-1, -1)`.
3. `seqlen >= ROCKE_SWAPQK_MIN_SEQLEN` (512) and `seqlen % block_n == 0`.
4. A kernel compiles for this `(Hq, Hk, D, causal, block_n, k_lds, gqa_fuse, v_paged)`.

Anything that fails routes to Triton. Kernels compile **lazily** on first use (~90 ms)
and are cached on the full config tuple, so an env flip can never serve a stale binary.

**Every gate fails silently.** That is the single most dangerous property of this
integration, and it burned us repeatedly (see [§7](#7-measurement-discipline)). The
backend therefore exports `ROCKE_STATS` counters — `swapqk`, `triton_slice`,
`fallback`, `paged_v`, `paged_v_declined` — and **every benchmark asserts on them.**
A run that got faster while dispatching zero swapqk kernels is not a result.

**Knobs**, all read once at import and folded into the cache key:
`ROCKE_K_LDS`, `ROCKE_V_TRANSPOSED`, `ROCKE_V_PAGED`, `ROCKE_BLOCK_N`,
`ROCKE_GQA_FUSE` (+ `_CAP`), `ROCKE_SWAPQK_MIN_SEQLEN`.

**A correctness constraint that shaped the design.** The KV loop bound is
`loop_stop = seqlen_k // block_n` — plain integer division, **the tail is truncated,
not masked**. So `seqlen_k % block_n == 0` is a *correctness* requirement, not a perf
preference. An early gate tested `seqlen % 32 != 0` while running a 64-wide tile, so a
2080-token request silently attended to only 2048 keys. Both the vLLM gate and the
in-repo torch op now share the same picker and reject anything non-divisible.

---

## 3. Isolated kernel results

Hq=32, Hk=8, D=128, causal, batch 1, fp16. `rocprofv3` dispatch duration, arms
interleaved A B A B, first rep discarded, **minimum across reps**.

### 3.1 The optimization chain

Session A — one interleaved session, all five arms, all five lengths. Times in ms.

| S | ① shipped default<br>`F=1, bn64, qk_douter=True` | ② `qk_douter=False` | ③ + `gqa_fuse=4` | Triton |
|---:|---:|---:|---:|---:|
| 1024 | 0.449 | 0.335 | **0.280** | 0.427 |
| 2048 | 3.924 | 2.320 | **1.199** | 1.853 |
| 3072 | 14.464 | 7.473 | **3.347** | 4.163 |
| 4096 | 30.088 | 15.671 | 7.983 | 7.108 |
| 8192 | 163.219 | 92.590 | 41.044 | 26.737 |

Session B — the `k_lds` head-to-head, one interleaved session. Times in µs.

| S | bn64 (= ③ above) | bn64 + `k_lds` | bn128 | Triton |
|---:|---:|---:|---:|---:|
| 1024 | 279.7 | **269.4** | 396.0 | 423.3 |
| 2048 | 1231.2 | **1112.8** | 1652.4 | 1795.4 |
| 4096 | 7484.4 | **4730.7** | 6579.7 | 6341.1 |
| 8192 | 37708.0 | **20936.7** | 28320.5 | 23945.4 |

### 3.2 Throughput, and why scaling was the real problem

| | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| swapqk (post-fusion, bn64) | 30.7 TF | 27.9 TF | 18.4 TF | **14.6 TF** |
| swapqk + `k_lds` | 31.9 TF | 30.9 TF | 29.1 TF | **26.3 TF** |
| Triton | 20.3 TF | 19.1 TF | 21.7 TF | 23.0 TF |

Read the *shape*, not the ratios. Going S=1024 → 8192 the work grows 64×, and:
bn64 grows **134.8×** (losing efficiency faster than it gains work), bn64+`k_lds` grows
**77.7×** (64× is the floor), Triton grows **56.6×** (it gets *more* efficient, because
its per-iteration cost is flat in S). `k_lds` does not make swapqk's scaling as good as
Triton's — it makes it good enough that swapqk's much lower constant factor is never
given back.

### 3.3 Final speedup vs Triton, shipped config

| S | swapqk + `k_lds` | Triton | speedup |
|---:|---:|---:|---:|
| 1024 | 269.4 µs | 423.3 µs | **1.57×** |
| 2048 | 1112.8 µs | 1795.4 µs | **1.61×** |
| 4096 | 4730.7 µs | 6341.1 µs | **1.34×** |
| 8192 | 20936.7 µs | 23945.4 µs | **1.14×** |

The `v_paged` change ([§5.4](#54-paged-v--reading-v-straight-out-of-the-kv-cache)) is
**kernel-neutral by design** — it costs +0.54%/layer in isolation (22703 → 22825 µs at
S=8192) and buys its entire win by deleting host-side work. It does not appear in this
table because it is not a kernel-throughput lever.

**Correctness** is gated on every timed run: `max_abs_diff = 4.883e-04`, `bad = 0` at
tolerance 2e-2, bit-identical across every config, `block_n`, and `k_lds`/`v_paged` arm.

---

## 4. End-to-end results

Qwen3-8B fp16, 36 layers, prefill TTFT. All arms **interleaved in one process**,
minimum of 3 reps, dispatch counters asserted, **all arms token-identical**.

- `k1v1` — `k_lds` on, V handed over as a per-layer `permute(1,2,0).contiguous()`
- `k1vp` — `k_lds` on, V gathered from the paged KV cache (the shipped default)
- `triton` — swapqk gate raised, same backend, same plumbing

**Graph mode** (ms, lower is better):

| S | k1v1 | k1vp | Triton | k1vp vs k1v1 | k1vp vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1024 | 862.3 | 863.6 | 872.3 | +0.15% | **−1.00%** |
| 2048 | 1664.4 | 1655.5 | 1684.2 | −0.53% | **−1.70%** |
| 4096 | 3302.3 | 3228.9 | 3281.2 | −2.22% | **−1.59%** |
| 8192 | 7049.2 | **6775.0** | 7022.0 | −3.89% | **−3.52%** |

**Eager mode** (ms):

| S | k1v1 | k1vp | Triton | k1vp vs k1v1 | k1vp vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1024 | 850.0 | 845.8 | 857.1 | −0.49% | **−1.32%** |
| 2048 | 1687.4 | 1679.0 | 1707.9 | −0.50% | **−1.69%** |
| 4096 | 3356.5 | 3270.5 | 3342.6 | −2.56% | **−2.16%** |
| 8192 | 7070.4 | **6817.1** | 7065.9 | −3.58% | **−3.52%** |

**Why 1.14× in the kernel becomes 3.5% end-to-end.** Attention is a *minority* of
prefill on this part. At S=2112 it is ~5% of the round (4 req × 36 layers × 2.35 ms
≈ 360 ms of 6500 ms), so even a 1.75× attention win predicts only +2.4% e2e — which is
exactly what the GQA-fusion e2e measurement showed (a null at S=2112, +7.8% median at
S=3968, matching a +7.4% arithmetic prediction). Attention is O(S²) while the rest of
the layer is O(S), so the e2e share — and the e2e win — grows with prompt length.

**Anyone optimizing e2e prefill at short prompts on this part should be looking at the
GEMMs, not at attention.**

Earlier e2e checkpoints, for the arc: `k_lds` alone was worth **~815–823 ms of a ~7.1 s
TTFT at S=8192 (11.5%)** — larger than the isolated benchmark predicted. Sequence-
adaptive `block_n` was worth **6.0%** at 8192-token prefill before `k_lds` retired it.

---

## 5. Optimizations that worked

### 5.1 `qk_douter = False` — flip one default

The shipped `SwapQKCfg` default was `qk_douter=True`, tuned at L=16384 / H=24 / D=128
MHA where it was documented as +3.3%. On the Qwen3-8B GQA shape it is a **loss
everywhere**, and the documented +3.3% does not reproduce even at its own shape:

| shape | `True` (shipped) | `False` | cost of the default |
|---|---:|---:|---:|
| Hq24 MHA S2048 dense | 19.68 TF | **20.66 TF** | −4.7% |
| Hq32/Hk8 GQA S2048 causal | 8.78 TF | **13.66 TF** | **−35.7%** |

Confirmed in `SQ_BUSY_CYCLES` with `SQ_WAVES` matched (identical launch geometry):
215.6M → 127.5M on the GQA shape, a **1.71×** penalty. In the S-sweep it is worth
1.34–1.94× depending on length.

The planned fix was "make it shape-aware." The measurement said the *sign* is not
shape-dependent — it loses at every shape tried. The fix was one line: flip the default.

### 5.2 GQA head fusion (`gqa_fuse=4`)

Process all F=4 query heads that share a KV head **in one CTA**, so each K/V tile is
fetched once and consumed four times. `block_size = 32 * n_waves * gqa_fuse` widens the
CTA to 256 threads while `q_rows_per_cta` stays fixed — the point is sharing, not more
work per CTA.

| S=2048 | F=1 | F=4 | Triton |
|---|---:|---:|---:|
| duration | 2354 µs | **1343 µs** (1.75×) | 1853 µs |
| `SQ_BUSY` | 126.4M | 65.4M (1.93×) | — |
| `TA_TA_BUSY` | 25.3M | 12.7M | 11.0M |
| VGPR / scratch | 216 / 0 | 216 / 0 | 256 / 168 B |

`SQ_WAVES` is identical across all arms — same launch geometry, so this is the kernel.
The gap to Triton on the texture-address path (the discriminator identified in the
whitepaper) closes from **2.30× to 1.16×**. It was thought this would need a d-blocked
epilogue to fit under the 256-VGPR wave32 ceiling; it did not — VGPR stayed at 216 with
zero scratch at F=1, 2 and 4.

**Fusion is a constant factor, not a scaling fix.** It is a 1.46×/1.40× win at
S=1024/2048 but a 0.82×/0.59× *loss* versus Triton at S=4096/8192 (crossover ≈ 3.3K),
because the super-linear growth survives it. That is what motivated §5.3.

Two things fell out of this work: a latent **causal-trim bug** (the kv-tile stop omitted
the `q_block` factor, silently dropping a kv block at `q_block>1` — a wrong answer, not
a slow one), and a **scheduler sign flip** — `sched_mode=pingpong` helps 7% when fused
and hurts 7% when unfused. A scheduling knob's sign is not invariant across a structural
change; re-sweep it, do not inherit it.

### 5.3 Staging K in LDS (`k_lds`)

The diagnosis first. At long S, swapqk was not limited by occupancy (24 waves/CU vs
Triton's 16), not by spills (swapqk spills nothing; **Triton** spills 168 B and wins
anyway), not by serialization (12.8% vs Triton's 26.4%, zero barriers), not by bank
conflicts (zero for both), and **not by DRAM bandwidth** — Triton sits at the ~102 GB/s
roofline while swapqk stalls 40% below it.

It was **L0 vector-cache thrash**. swapqk used no LDS by design, so every WMMA operand
came straight from the memory hierarchy and reuse depended entirely on the 32 KB L0.
The cleanest evidence: three configs at S=8192 executing an **identical** number of
vmem load instructions (33.8M / 33.8M / 34.1M) but differing **5.2× in L2 requests and
3.1× in wall time**.

The fix stages **only K** in LDS; V stays on `v_transposed` + `buffer_gather`. Per tile:

```
s_barrier                 # previous tile's readers are done
coop load K -> LDS        # whole CTA, 4 chunks/thread, global -> VGPR -> ds_write
s_barrier                 # tile visible to all waves
QK:  ds_read K, WMMA      # 2x ds_read_b128 + vec_concat per fragment
PV:  buffer_gather V      # UNCHANGED
```

Result at S=8192: total L2 requests fall **59.0M → 17.5M (3.37×)** for identical work,
and wall time follows at **2.07×**. `GL2C_MISS` falls 3.2× alongside `GL2C_HIT` — the CU
stops *asking* L2 for data it already has. A pre-registered kill criterion (≥2.4× fewer
L2 requests **and** ≥1.15× wall time, else the thesis is falsified) cleared with margin.

Cost: **+1 VGPR, 0 scratch, 17 KB LDS.** Instruction count goes *up* 7.4% — this is a
composition win, not an instruction-count win, and must not be presented as one. vmem
instructions drop 2.7× and the per-CTA K request count drops 512 → 32 for the same
16 KB of unique bytes, because half of the old K requests were pure address-pipe waste
(the A-operand row is `lane % 16`, so lanes *l* and *l+16* issue identical addresses).

The bank pad `_KVPAD = 8` is **derived, not tuned**: row stride `(128+8)` f16 = 68
dwords, `68 mod 32 = 4`, so the 8 lanes serviced per `ds_read_b128` pass tile all 32
banks exactly once. Measured `SQC_LDS_BANK_CONFLICT = 0` against 287M `LDS_IDX_ACTIVE`.

Two consequences beyond the speedup: swapqk now beats Triton at **every** measured
length — the 3.3K crossover is gone — and `block_n=128` is **retired** ([§6.2](#62-block_n128-shipped-then-deleted)).

### 5.4 Paged V — reading V straight out of the KV cache

After `k_lds`, swapqk won 1.14× in isolation but only **tied** Triton end-to-end. The
whole gap was one line in the backend:

```python
v_t = v.permute(1, 2, 0).contiguous() if cfg.v_transposed else v
```

The kernel's `P×V` A-fragment wants V as `[Hk, D, Sk]` — token fastest-varying — so the
backend materialised a transposed copy **every layer, every forward**. Measured cost:

| S | ms/layer | GB/s | MB of V | ms/forward (36 layers) |
|---:|---:|---:|---:|---:|
| 1024 | 0.088 | 47.4 | 2.10 | 3.2 |
| 2048 | 0.413 | 20.3 | 4.19 | 14.9 |
| 4096 | 1.950 | 8.6 | 8.39 | 70.2 |
| 8192 | **5.798** | **5.8** | 16.78 | **208.7** |

Note the bandwidth *degrades* 47.4 → 5.8 GB/s as S grows — ~18× below what the fabric
can do — because the gather stride is `Sk` elements. That is ~209 ms of a ~7.1 s TTFT.

**The insight: the transpose is already done.** vLLM's paged V cache is *already stored
transposed*. `PagedAttention.split_kv_cache` (from `vllm.v1.attention.ops.paged_attn`)
yields `[num_blocks, num_kv_heads, head_size, block_size]` — token fastest-varying,
exactly the order the A-fragment wants. `reshape_and_cache` performs the transpose
during `unified_kv_cache_update`, which runs *before* attention with an explicit data
dependency. **We were paying 5.8 ms/layer to redo work the cache write already did.**

New kernel knobs `v_paged` + `kv_block_size` gather V through the block table:

```
element offset = blk*(Hk*D*bs) + kv_head*(D*bs) + d_col*bs + tok
  voffset = d_col * bs * 2                                  (per-lane, VGPR)
  soffset = (blk*(Hk*D*bs) + kv_head*(D*bs) + tok) * 2      (uniform, SGPR)
```

Two facts made this cheap rather than invasive. `block_size` is always a multiple of 16,
so a 16-key WMMA A-fragment **never straddles a physical block** — the gather keeps its
exact 2× `dwordx4` shape. And paging is *better behaved* than the old addressing: the
per-lane term shrinks from `d_col*seqlen_k` (up to 2.08 MB, a runtime multiply) to
`d_col*bs*2` (≤ 4064 B, a compile-time-constant multiply).

**Result:** the gather costs **+0.54%/layer** (22703 → 22825 µs at S=8192, interleaved,
3 reps) against ~209 ms of permute deleted. End-to-end that is the **−3.5% vs Triton**
in [§4](#4-end-to-end-results), with **no crossover** — neutral at short prompts, and
the win grows with S because the permute is O(S). Shipped **on by default**
(`ROCKE_V_PAGED=1`) after the sweep. GPU numerics: 12/12 PASS over
B ∈ {1,2} × S × `block_size` ∈ {16,32,64} with shuffled, poison-filled pages.

Two traps this exposed, both worth remembering:

- **The paged gates fail silently into the *permute path*, not into Triton**, so the
  `swapqk` counter cannot see them. `ROCKE_STATS` gained `paged_v` /
  `paged_v_declined` and the harness asserts on them — the very first smoke run was a
  silent fallback that only the assertion caught.
- **`to_sgpr_u32` (readfirstlane) on the raw block id is mandatory, not an
  optimization.** AMDGPU treats every `addrspace(1)` load as divergent, so without it
  the backend wraps every V load in a **32-iteration waterfall loop**. It is invisible
  in the load counts — only the `readfirstlane` count guards it, which is why there is a
  CPU-only regression test on exactly that count.

---

## 6. Optimizations that were expected to help and did not

### 6.1 `v_transposed = False` — remove the permute the obvious way

The natural way to kill the §5.4 permute: tell the kernel to consume row-major V. It is
a **wash**, and it was measured *before* the paged idea existed.

The kernel falls back to the row-major V gather — **16× `buffer_load_f16_d16` per
fragment instead of 2× `dwordx4`** — and the two costs nearly cancel, with a crossover
around S=4096:

| S | e2e delta vs `v_transposed=True` (graph / eager) |
|---:|---|
| 1024 | ~0 / ~0 |
| 2048 | −4 / −7 ms |
| 4096 | −46 / −49 ms (row-major wins) |
| 8192 | **+67 / +50 ms** (row-major loses) |

Kernel-only A/B confirms the mechanism: at S=8192 row-major costs **+8.34 ms/layer**
(21921 → 30259 µs, 25.08 → 18.17 TF) ≈ +300 ms over 36 layers, against the ~209 ms of
permute it saves.

**The lesson:** the problem was never the transpose's *existence*, it was its
*implementation*. `v_paged` gets the permute deleted **and** keeps the fast gather,
which is why it has no crossover and this does.

### 6.2 `block_n=128` — shipped, then deleted

Widening the KV tile buys L0 hits (L2 requests per texture load 1.55 → 0.95), so it is
a 13–29% win above S=4096 and a 18–32% loss below it. It shipped behind a measured
sequence-adaptive lookup table and was worth **6.0% e2e** at 8192-token prefill.

**`k_lds` retired it the same day.** LDS buys strictly more L0 relief without bn128's
112 B of scratch: bn64+`k_lds` beats bn128 by 1.39× at S=4096 and 1.35× at S=8192, and
bn128 already lost below that. The adaptive table collapsed to **one row** — which also
*widens* eligibility, since bn64 divides every length bn128 did and more, so fewer
requests fall through to Triton.

`k_lds` and `block_n=128` **compete, they do not stack**: a padded K tile is 17 KB at
bn64 (3 workgroups still fit the 64 KB per-CU LDS, occupancy stays at 24 waves/CU) but
35 KB at bn128, which collapses occupancy to **1 WG/CU**.

`block_n=256` spills 632 B and loses everywhere. Not a candidate.

### 6.3 `kv_lds` — staging K **and** V in LDS

The first attempt at the LDS idea measured **0.38× / 0.35× / 0.29×** at
L=512/2048/4096 — losing worse as L grew, the exact opposite of the hypothesis. It was
recorded as dead. Three root causes, and why the K-only retry was different:

| recorded cause | why K-only at `gqa_fuse=4` differed |
|---|---|
| VGPR 197→256 plus 16 B spill | The coop loader's live payload. `kv_lds` staged K+V across **64** threads = 128 VGPR; K-only across a **256**-thread CTA = 16 VGPR, and the chunk row/col div+mod hoists out of the loop. Measured: **+1 VGPR, 0 scratch.** |
| `ds_load` 0 → 320 | Entirely a **V** problem — the flat LDS V read is 16 uncoalesced scalar `ds_load`s per fragment. Staging K only never touches it. |
| 2 barriers/tile kill pingpong | Still real in principle, and now a harder 8-wave rendezvous. Measured: it did not materialise — pingpong stayed the better arm for both configs at both lengths. |

**The lesson is not "LDS works after all."** The earlier verdict was *correct for its
configuration* (H24 dense, `gqa_fuse=1`, 2 waves sharing the tile) and was invalidated
by an unrelated change — head fusion made the amortisation 4× larger — that nobody
re-checked it against. Negative results have a configuration attached.

### 6.4 `head_adjacent` dispatch swizzle — a prediction that failed, twice over

A falsification probe before building fusion: reorder the dispatch so the F CTAs sharing
a KV head are adjacent in flight. The plan predicted a **null**, with arithmetic (at
216 VGPR the concurrent window already spans ~2 KV heads ≈ 2 MB ≈ L2, so they are
already co-temporal).

**The prediction was wrong** — the swizzle measured **1.27×** (`SQ_BUSY` 118.8M →
93.7M). The K/V refetch cost is *partly* temporal and reordering recovers that part; the
larger part is that the sharing CTAs do not share a **CU**, and only fusion recovers
that. It was **not shipped** anyway: fusion strictly dominates and subsumes it, so a
second dispatch-order knob would be cost without benefit.

### 6.5 `q_block=2` — a "verified 2.05× win" that was a confound

The kernel hard-rejects `qk_douter and q_block > 1`, so **every** `q_block=2` build
silently disabled `qk_douter` — and that was the entire effect. Held constant,
`q_block=2` is *worse* at S=2048 (163.5M vs 127.2M `SQ_BUSY`, and it spills 624 B).

### 6.6 The knob sweep that found nothing

At S=4096 on top of `block_n=32, qk_douter=False` (900.6M `SQ_BUSY` baseline), none of
these beat −7%: `o_nt` 894.1 · `prefetch_v` 909.8 · `v_prefetch=2` 912.6 ·
`bcast_group=4` 984.9 · `o_f16` 1002.9 · `qk_ilp=4` 1426.5 · `qk_ilp=1` 1783.7.

Knob tuning was exhausted here. Everything that worked afterwards
(§5.1–§5.4) was structural.

### 6.7 The "~8% rocKE plumbing win" — real, but not ours

An early round reported the rocKE backend ~8% faster than vLLM's default **while
dispatching zero swapqk kernels**, attributed to rocKE's plumbing.

The effect is real (**7.10%** mean, 6/6 interleaved pairs, thermals flat, output
bit-identical through 64 decode tokens). The attribution was wrong. vLLM contains **two
different functions named `context_attention_fwd`**:

| | `ops/prefix_prefill.py` | `ops/triton_prefill_attention.py` |
|---|---|---|
| kind | **paged** — block-table indirection, cached-context merge, fp8 dequant | **dense** — contiguous `key`/`value` |
| reached by | `RocmAttentionImpl.forward` (vLLM's default) | `RockeAttentionImpl._forward_rocke` |

The paged kernel performs its indirection and cached-context merge **even when there is
zero cached context**, which is exactly a cold full prefill. That is the entire gap.
Measured on its own, the rocKE backend class contributes **nothing** — it is marginally
*slower*. The win belongs upstream in `RocmAttentionImpl`, not in a rocKE file, and its
addressable market shrinks once prefix caching or chunked prefill is on.

### 6.8 Two other retractions worth carrying

- **"swapqk is a +4.8% e2e win"** — measured against the wrong control (vLLM's default
  backend, which changes plumbing, KV handling and kernel at once). Against a matched
  same-backend control it was a **3.4% regression** at the time.
- **"graph mode is ~1.05 s faster than eager"** — an artifact of comparing two separate
  process launches. In matched within-process runs they are within noise (7062 vs 7085
  at S=8192). Consistent with `splitting_ops = attention_ops`: attention is split *out*
  of piecewise cudagraphs, so at prefill sizes launch overhead is negligible.

---

## 7. Measurement discipline

Roughly half the elapsed effort on this project went into **measurement defects that
produced confident, plausible, wrong numbers** — several of which had already been
written up as conclusions. The rules that came out of it:

1. **Confirm the kernel actually dispatched.** vLLM's EngineCore is *forked* and
   inherits the parent's mutated `sys.path`; one harness line stripping `rocke-repo`
   from `sys.path` made **every** early e2e number Triton-vs-Triton. Assert on
   `ROCKE_STATS` every time.
2. **Confirm the work was actually performed.** vLLM v1 enables **prefix caching by
   default**, so `[PROMPT] * 4` computes the prefill once and replays it. The harness
   reported 18,339 tok/s; the real figure was 1,189.
3. **Interleave A B A B — a blocked A/B cannot separate the arm from its slot**, and
   running the order forwards then backwards does *not* fix it if an arm keeps landing
   in an extreme slot.
4. **Take the minimum across reps**, never the median, for anything over ~5 ms. The box
   is shared; contention can only inflate. It also up-clocks 609 → 938 MHz as it warms,
   so discard the first rep.
5. **Never stitch a table together from probes run hours apart** — inter-session clock
   drift is ~13%.
6. **Never compare arms across separate process launches** — run-to-run drift is ~1 s at
   S=8192, which swamps the effects being measured.
7. **`max_tokens=1` cannot validate a prefill change.** It compares the prefill output
   tensor and never reads the KV cache back. A path that computes the right output but
   corrupts the cache passes it. Use a 64-token decode hash.
8. **GRBM counters are broken on this part** — `GRBM_COUNT` reads a fixed ~1.0M
   regardless of duration, i.e. *below* `GRBM_GUI_ACTIVE`. Use `SQ_BUSY_CYCLES` /
   `SQ_WAVE_CYCLES`. `MemUnitBusy` is also non-functional (returns 319, 656 for a
   documented 0–100 percentage), and gfx1151 exposes **no stall counters at all**.
9. **`rocprofv3` multiplexes counters across dispatches.** One counter per pass, and
   report coverage.
10. **Do not profile the correctness oracle.** A torch/rocBLAS reference emits
    dispatches *larger* than the kernel under test, so a "biggest kernel" heuristic
    silently selects `Cijk_...` — every early Triton number was measuring our own
    reference GEMM.
11. **Verify two arms calling "the same function" resolve to the same module**
    (see §6.7).
12. **Log thermals rather than assuming them.** "It's probably thermal" is easy and
    unfalsifiable; sampling `rocm-smi --showtemp --showpower --showclocks` between arms
    costs nothing and turns it into data. (It was 32–38 °C and the effect was real.)

Operational traps: a killed benchmark orphans a `VLLM::EngineCore` holding ~88 GB of the
unified pool — it has PPID 1 and comm `VLLM::EngineCor`, so it does *not* match
`pgrep -f <script>`; find it with `rocm-smi --showpids` and kill by PID. And
**`pkill -f <pattern>` over SSH matches your own remote shell** and kills the session
(exit 255) before it kills the target.

---

## 8. Document index

| Document | What it covers |
|---|---|
| [`swapqk_vs_triton_whitepaper.md`](../../../../docs/swapqk_vs_triton_whitepaper.md) | The SwapQK algorithm, the WMMA fragment layout, the original gap analysis, and §0's retractions |
| [`status_plus_next_steps_08_11_2026.md`](../../../../docs/status_plus_next_steps_08_11_2026.md) | First operational record: the seven measurement defects, the 3.4% e2e regression, the priority list |
| [`readme_repro_08_12_2026.md`](../../../../docs/readme_repro_08_12_2026.md) | Reproducing the published numbers — which reproduced, which did not, and the `qk_douter` verdict (§5.1) |
| [`rocke_backend_impact_08_12_2026.md`](../../../../docs/rocke_backend_impact_08_12_2026.md) | The dense-vs-paged Triton finding (§6.7), in full |
| [`status_plus_next_steps_08_12_2026.md`](../../../../docs/status_plus_next_steps_08_12_2026.md) | Second operational record; P0 closed |
| [`gqa_head_fusion_08_16_2026.md`](../../../../docs/gqa_head_fusion_08_16_2026.md) | GQA head fusion (§5.2), the causal-trim bug, the scheduler sign flip, the swizzle probe |
| [`long_seq_scaling_08_17_2026.md`](../../../../docs/long_seq_scaling_08_17_2026.md) | Why swapqk fell behind past ~3.3K: the L0 thrash diagnosis, with occupancy/spills/LDS/bandwidth all ruled out |
| [`adaptive_block_n_08_18_2026_not_necessary.md`](../../../../docs/adaptive_block_n_08_18_2026_not_necessary.md) | Sequence-adaptive `block_n` (§6.2) — shipped, then superseded the same day |
| [`k_lds_staging_08_18_2026_v02.md`](../../../../docs/k_lds_staging_08_18_2026_v02.md) | Staging K in LDS (§5.3), and why the same lever lost 3× the first time |
| [`paged_v_gather_08_19_2026.md`](../../../../docs/paged_v_gather_08_19_2026.md) | Paged V (§5.4), the full e2e sweep, and the waterfall trap |

---

## 9. Reproducing

Everything runs on the gfx1151 box. vLLM is a **source checkout on `PYTHONPATH`**, not
pip-installed.

```bash
source ~/.pyenv/versions/3.11.9/envs/rocm-env/bin/activate
R=<repo>/dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH=~/vllm:$R/library:$R/platform/python
export ROCKE_CPP_QUIET_FALLBACK=1
cd $R/library
```

**CPU-only tests** (70 cases, no GPU — these run anywhere):

```bash
python -m pytest tests/test_gfx1151_wmma_fmha_swapqk.py
```

They cover the `is_valid_spec` rejection matrix, kernel-name/artifact-cache collisions,
the ABI param list (paged adds exactly `VBlockTable` + `bt_stride`, and only when
`v_paged`), the LDS allocation size, the `readfirstlane` waterfall guard, and the
2× `dwordx4` V load shape.

**Correctness + throughput gate:**

```bash
python -m builders.gfx1151.attention.wmma_fmha_swapqk_verify \
    --batch 1 --seqlen-q 8192 --seqlen-k 8192 --causal \
    --heads 32 --kv-heads 8 --head-size 128 \
    --gqa-fuse 4 --block-n 64 --k-lds \
    --v-paged --kv-block-size 16 --shuffle-blocks 1
```

The paged path **poison-fills** the fake cache with `7777.0` before scattering and
shuffles the block table by default, so an addressing bug screams instead of returning
plausibly-clamped zeros. The CPU reference keeps consuming the original row-major V —
the test is that the paged kernel reproduces it.

**End-to-end** (`~/e2e_qwen.py`, arms interleaved in one process, graph mode first then
eager, `ROCKE_STATS` asserted):

```bash
python3 ~/e2e_qwen.py --arms k1v1,k1vp,triton --lens 1024,2048,4096,8192 --reps 3
```

Before any sweep: `uptime` and `rocm-smi --showpids`. If the latter is not *"No KFD PIDs
currently running"*, an orphan is holding ~88 GB and the run will die at startup.

---

## 10. Known gaps

- **`k_lds` is a bn64-only lever**, by LDS budget. Only D=128 has been timed, only
  S ≤ 8192, only causal, batch 1, Hq32/Hk8.
- **Single-buffered by choice.** Double-buffering K would hide the staging latency
  behind the previous tile's WMMA at the cost of 35 KB and the occupancy cliff. Not
  attempted, not measured.
- **The `_SWAPQK_MIN_SEQLEN = 512` floor is plausibly too conservative** — swapqk now
  beats Triton by 1.57× at S=1024, but nothing below S=1024 has been measured.
- **The truncating tail bound is unchanged.** `seqlen_k % block_n != 0` still routes to
  Triton. Paging does not fix it and the two must not be conflated.
- **2 GiB SRD cap.** The paged `soffset` is a 32-bit byte offset; per-layer V cache is
  ~1 GiB at the current 70 GiB total, so it fits, but not with margin. Guarded
  host-side rather than by building an i64 path.
- **The in-repo torch custom op never sets `gqa_fuse`**, so it compiles at F=1 while the
  vLLM backend picks F=4. It also does a device→host→device round trip for the V
  transpose. Both pre-existing, both worth fixing separately.
