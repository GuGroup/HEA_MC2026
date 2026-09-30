#!/usr/bin/env python3
"""Copy a UQMC results root to a separate destination tree.

The stored activity values in this project are already in log-domain, so the
right comparison is between the existing unscaled activity columns and the
existing [-1, 0] scaled columns. This helper just mirrors the shard layout and
activity CSVs so the two modes can be run side by side without overwriting one
another.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES = (
    "predicted_activity_selected_by_trial.csv",
    "predicted_activity_all_snapshots_by_trial_temperature.csv",
    "selected_temperature_by_trial.csv",
    "best_temperature_by_trial_crps.csv",
    "random_crps_by_trial.csv",
    "crps_by_trial_temperature.csv",
    "best_trial_cemc_crps.csv",
    "best_trial_random_crps.csv",
    "cemc_global_candidates.csv",
    "random_global_candidates.csv",
)


def copy_root(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        raise SystemExit(f"Source does not exist: {src_root}")
    if dst_root.exists():
        raise SystemExit(f"Destination already exists: {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=False)
    for shard in sorted(src_root.glob("shard_*")):
        if not shard.is_dir():
            continue
        dst_shard = dst_root / shard.name
        dst_shard.mkdir(parents=True, exist_ok=False)
        for name in FILES:
            src = shard / name
            if src.exists():
                shutil.copy2(src, dst_shard / name)
        for extra in ("completed_trials.txt",):
            src = shard / extra
            if src.exists():
                shutil.copy2(src, dst_shard / extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-results-root", default="results", help="Original results root")
    parser.add_argument("--dst-results-root", required=True, help="Destination mirror root")
    args = parser.parse_args()
    copy_root(Path(args.src_results_root).resolve(), Path(args.dst_results_root).resolve())


if __name__ == "__main__":
    main()
