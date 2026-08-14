#!/usr/bin/env python3
# analyze_lds_k_group_pad.py — aggregate sweep_lds_k_group_pad.sh results and
# produce a summary table comparing lds_k_group_pad values across shapes.
#
# Usage:
#   python analyze_lds_k_group_pad.py                          # default /tmp/dense_kpad_sweep/
#   python analyze_lds_k_group_pad.py --input-dir /data/sweeps
#   python analyze_lds_k_group_pad.py --input-dir /tmp/dense_kpad_sweep/ --csv out.csv

import argparse
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument(
        "--input-dir",
        default="/tmp/dense_kpad_sweep",
        help="Directory containing JSON files from sweep_lds_k_group_pad.sh",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional path to write the full pivot table as CSV",
    )
    ap.add_argument(
        "--reference-pad",
        type=int,
        default=8,
        help="Pad value used as the baseline for relative-gain columns (default: 8)",
    )
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas", file=sys.stderr)
        return 1

    pattern = str(Path(args.input_dir) / "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No JSON files found in {args.input_dir}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        try:
            rows.extend(json.load(open(f)))
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)

    if not rows:
        print("No records loaded.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    n_total = len(df)
    df = df[df["ok"] == True]  # noqa: E712
    n_ok = len(df)
    print(
        f"Loaded {n_total} records, {n_ok} passed correctness check ({n_total - n_ok} dropped).\n"
    )

    if df.empty:
        print("No passing records to analyse.", file=sys.stderr)
        return 1

    # Ensure lds_k_group_pad is numeric (the JSON may store it as int already).
    df["lds_k_group_pad"] = pd.to_numeric(df["lds_k_group_pad"])

    key_cols = ["mode", "label", "Hq", "Hkv", "block_n", "sliding_window"]
    # Guard against missing columns from older JSON formats.
    key_cols = [c for c in key_cols if c in df.columns]

    pivot = df.pivot_table(
        index=key_cols,
        columns="lds_k_group_pad",
        values="tflops",
        aggfunc="mean",
    )

    ref = args.reference_pad
    pads = sorted(pivot.columns.tolist())

    pivot["best_pad"] = pivot.idxmax(axis=1)

    for pad in pads:
        if pad != ref and ref in pivot.columns and pad in pivot.columns:
            col = f"gain_vs_{ref}_pad{pad}"
            pivot[col] = (pivot[pad] - pivot[ref]) / pivot[ref] * 100

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    gain_cols = [c for c in pivot.columns if c.startswith("gain_vs_")]
    display_cols = ["best_pad"] + gain_cols + pads

    print("=" * 80)
    print("TFLOPS pivot (rows = shape, columns = lds_k_group_pad)")
    print(f"Gain columns show % change relative to pad={ref}")
    print("=" * 80)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.3f}".format,
    ):
        print(pivot[display_cols].to_string())

    # ------------------------------------------------------------------ #
    # Best-pad distribution
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("Best pad distribution across all shapes:")
    print("=" * 80)
    print(pivot["best_pad"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------------ #
    # Shapes where a pad other than the reference wins by > 1%
    # ------------------------------------------------------------------ #
    threshold = 1.0
    print(f"\n{'=' * 80}")
    print(f"Shapes where best_pad != {ref} AND gain > {threshold}%:")
    print("=" * 80)
    winners = pivot[pivot["best_pad"] != ref].copy()
    if winners.empty:
        print(f"  None — pad={ref} is best (or tied) on every shape.")
    else:
        # Show only rows where at least one gain col exceeds the threshold.
        mask = False
        for col in gain_cols:
            mask = mask | (winners[col].abs() > threshold)
        significant = winners[mask]
        if significant.empty:
            print(f"  No shape shows a gain > {threshold}% vs pad={ref}.")
        else:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                200,
                "display.float_format",
                "{:.3f}".format,
            ):
                print(significant[display_cols].to_string())

    # ------------------------------------------------------------------ #
    # Optional CSV output
    # ------------------------------------------------------------------ #
    if args.csv:
        pivot.to_csv(args.csv)
        print(f"\nFull pivot table written to: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
