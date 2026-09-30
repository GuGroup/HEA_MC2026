#!/usr/bin/env python3
"""Concatenate completed shard CSV outputs into results/combined."""
from __future__ import annotations

import argparse
from pathlib import Path


CSV_FILES = [
    "predicted_activity_selected_by_trial.csv",
    "predicted_activity_all_snapshots_by_trial_temperature.csv",
    "trial_composition_samples.csv",
    "random_slab_seeds.csv",
    "metrics_by_trial_temperature.csv",
    "random_metrics_by_trial.csv",
    "selected_temperature_by_trial.csv",
    "trial_timing.csv",
]


def concat_csv(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output.open("w") as out:
        for path in inputs:
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open() as f:
                header = f.readline()
                if not header:
                    continue
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in f:
                    out.write(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--output", default="results/combined")
    args = p.parse_args()

    root = Path(args.results_root)
    outdir = Path(args.output)
    shards = [root / f"shard_{idx:02d}" for idx in range(args.world_size)]
    for name in CSV_FILES:
        concat_csv([shard / name for shard in shards], outdir / name)
    print(f"Wrote combined CSVs under {outdir}")


if __name__ == "__main__":
    main()
