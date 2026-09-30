#!/usr/bin/env python3
"""Collect v1.0 plot_data CSVs from uq_ML1_full...uq_ML6_full.

Usage:
  python scripts/collect_plot_data.py --root . --output all_ml_plot_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KNOWN_FILES = [
    "tau_mse_scatter_by_trial.csv",
    "paired_delta_by_trial.csv",
    "method_metric_interval_summary.csv",
    "paired_delta_interval_summary.csv",
    "cemc_temperature_metric_by_trial.csv",
    "temperature_metric_summary.csv",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Directory containing uq_ML*_full workdirs")
    p.add_argument("--output", default="all_ml_plot_data")
    p.add_argument("--mls", default="1,2,3,4,5,6")
    args = p.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    mls = [int(x) for x in args.mls.split(",") if x.strip()]

    for name in KNOWN_FILES:
        frames = []
        for ml in mls:
            path = root / f"uq_ML{ml}_full" / "results" / "plot_data" / name
            if not path.exists():
                print(f"Missing {path}; skipping for {name}")
                continue
            df = pd.read_csv(path)
            if "ml" in df.columns:
                df["ml"] = ml
            else:
                df.insert(0, "ml", ml)
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(outdir / name, index=False)
            print(f"Wrote {outdir / name}: {len(combined)} rows")


if __name__ == "__main__":
    main()
