#!/usr/bin/env bash
# sweep_lds_k_group_pad.sh — sweep lds_k_group_pad for the gfx950 attention_dense D=64 kernel.
#
# lds_k_group_pad is only live at D=64 (at D=128 the kernel uses a per-row pad
# instead and ignores the group pad). Sweeps:
#   pads    : 0 8 16 24 32
#   block_n : 64 128
#   GQA     : Hq=128/Hkv=8, Hq=64/Hkv=8, Hq=32/Hkv=32 (MHA)
#   modes   : causal (S=2048/4096/8192), swa, varlen, persistent
#
# Results land in OUT_DIR (default /tmp/dense_kpad_sweep/) as one JSON per
# (pad, block_n, Hq, Hkv) combination. Pass them to analyze_lds_k_group_pad.py
# to produce the summary table.
#
# Usage (from repo root or rocke/platform/):
#   bash rocke/library/benchmarks/gfx950/attention/prefill/sweep_lds_k_group_pad.sh
#   OUT_DIR=/data/sweeps bash ... sweep_lds_k_group_pad.sh
#   WARMUP=5 ITERS=20 bash ... sweep_lds_k_group_pad.sh   # quick smoke run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/benchmark_dense_prefill_live.py"

OUT_DIR="${OUT_DIR:-/tmp/dense_kpad_sweep}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
SEED="${SEED:-0}"

PADS=(0 8 16 24 32)
BNS=(64 128)
# "Hq Hkv" pairs: GQA-16, GQA-8, MHA
HQ_HKV_PAIRS=("128 8" "64 8" "32 32")

mkdir -p "${OUT_DIR}"

echo "Output dir : ${OUT_DIR}"
echo "Warmup     : ${WARMUP}  Iterations: ${ITERS}"
echo "Pads       : ${PADS[*]}"
echo "block_n    : ${BNS[*]}"
echo "Hq/Hkv     : ${HQ_HKV_PAIRS[*]}"
echo ""

total=$(( ${#PADS[@]} * ${#BNS[@]} * ${#HQ_HKV_PAIRS[@]} ))
run=0

for pad in "${PADS[@]}"; do
    for bn in "${BNS[@]}"; do
        for pair in "${HQ_HKV_PAIRS[@]}"; do
            hq=$(echo "$pair" | cut -d' ' -f1)
            hkv=$(echo "$pair" | cut -d' ' -f2)
            run=$(( run + 1 ))
            out="${OUT_DIR}/pad${pad}_bn${bn}_hq${hq}_hkv${hkv}.json"
            echo "[${run}/${total}] pad=${pad} bn=${bn} Hq=${hq} Hkv=${hkv}  ->  ${out}"
            python "${BENCHMARK}" \
                --mode all \
                --d 64 \
                --bn "${bn}" \
                --hq "${hq}" \
                --hkv "${hkv}" \
                --lds-k-group-pad "${pad}" \
                --warmup "${WARMUP}" \
                --iterations "${ITERS}" \
                --seed "${SEED}" \
                --output-json "${out}"
        done
    done
done

echo ""
echo "Sweep complete. JSON results in: ${OUT_DIR}"
echo "Run analyze_lds_k_group_pad.py to summarise:"
echo "  python ${SCRIPT_DIR}/analyze_lds_k_group_pad.py --input-dir ${OUT_DIR}"
