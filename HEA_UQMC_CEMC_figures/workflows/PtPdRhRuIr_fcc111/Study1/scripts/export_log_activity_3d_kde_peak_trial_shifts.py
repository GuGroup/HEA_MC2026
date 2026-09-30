#!/usr/bin/env python3
"""Export BE and composition shifts for the log-activity 3D-KDE peak trial."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def compile_cpp(source: Path, output: Path) -> None:
    subprocess.run(
        ["g++", "-O2", "-std=c++11", str(source), "-o", str(output)],
        check=True,
    )


def select_peak(points: pd.DataFrame) -> pd.Series:
    required = {"trial", "temperature", "tau", "mse", "crps", "kde_probability"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"KDE input is missing columns: {sorted(missing)}")
    clean = points.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    candidates = clean[clean["tau"] > 0]
    if candidates.empty:
        raise ValueError("No log-activity 3D-KDE point has tau > 0")
    return candidates.loc[candidates["kde_probability"].idxmax()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kde-input", type=Path,
        default=Path("results/kde_neighbor_be_temperature_log/log_activity_kde_peak_nearest_trials.csv"),
    )
    parser.add_argument(
        "--be-exporter", type=Path,
        default=Path("scripts/export_trial_be_shifts_log_activity.cpp"),
    )
    parser.add_argument(
        "--composition-exporter", type=Path,
        default=Path("scripts/export_neighbor_composition_shifts_log_activity.cpp"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/log_activity_3d_kde_peak_trial_shifts"),
    )
    parser.add_argument("--n-compositions", type=int, default=1400)
    parser.add_argument("--random-seed", type=int, default=20260706)
    parser.add_argument("--be-mean", type=float, default=0.04233)
    parser.add_argument("--be-sigma", type=float, default=0.2604)
    parser.add_argument("--comp-mean", type=float, default=0.000347)
    parser.add_argument("--comp-sigma", type=float, default=0.046858)
    args = parser.parse_args()

    points = pd.read_csv(args.kde_input)
    peak = select_peak(points)
    peak_trial = int(peak["trial"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / "log_activity_3d_kde_peak_trial.csv"
    metadata_columns = [
        column for column in
        ("trial", "temperature", "tau", "mse", "crps", "kde_density", "kde_probability")
        if column in peak.index
    ]
    peak[metadata_columns].to_frame().T.to_csv(metadata_path, index=False)

    be_executable = args.output_dir / "export_trial_be_shifts_log_activity"
    all_be_path = args.output_dir / "_be_shifts_through_peak_trial.csv"
    compile_cpp(args.be_exporter, be_executable)
    with all_be_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [str(be_executable), str(peak_trial + 1), str(args.random_seed),
             str(args.be_mean), str(args.be_sigma)],
            stdout=handle, check=True,
        )
    be = pd.read_csv(all_be_path)
    peak_be = be.loc[be["trial"] == peak_trial]
    if len(peak_be) != 1:
        raise ValueError(f"Expected one BE-shift row for trial {peak_trial}, got {len(peak_be)}")
    be_path = args.output_dir / "log_activity_3d_kde_peak_trial_be_shifts.csv"
    peak_be.to_csv(be_path, index=False)
    all_be_path.unlink()

    trial_list_path = args.output_dir / "_peak_trial.csv"
    pd.DataFrame({"trial": [peak_trial]}).to_csv(trial_list_path, index=False)
    composition_executable = args.output_dir / "export_neighbor_composition_shifts_log_activity"
    composition_path = args.output_dir / "log_activity_3d_kde_peak_trial_composition_shifts.csv"
    compile_cpp(args.composition_exporter, composition_executable)
    subprocess.run(
        [str(composition_executable), str(trial_list_path), str(args.n_compositions),
         str(args.random_seed), str(args.comp_mean), str(args.comp_sigma), str(composition_path)],
        check=True,
    )
    trial_list_path.unlink()

    composition = pd.read_csv(composition_path)
    if len(composition) != args.n_compositions or composition["trial"].nunique() != 1:
        raise ValueError("Composition-shift output does not contain exactly one complete trial")

    print(f"Log-activity 3D-KDE peak: trial {peak_trial}, T={int(peak['temperature'])} K")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {be_path} (1 row)")
    print(f"Wrote {composition_path} ({len(composition)} rows)")


if __name__ == "__main__":
    main()
