#!/usr/bin/env python3
"""Export trial-aligned UQMC CEMC temperature metrics and random slab metrics.

Outputs, relative to the UQMC project directory by default:
- cemc_temperature_metric_by_trial.csv
- paired_delta_by_trial.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEMPERATURES = list(range(2000, 299, -100))
N_TRIALS = 10_000
METRICS = ["tau", "mse"]


def _expected_trials() -> set[int]:
    return set(range(N_TRIALS))


def _load_shard_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shard_dirs = sorted(p for p in results_dir.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under {results_dir}")

    cemc_frames: list[pd.DataFrame] = []
    random_frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        metrics_path = shard_dir / "metrics_by_trial_temperature.csv"
        random_path = shard_dir / "random_metrics_by_trial.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        if not random_path.exists():
            raise FileNotFoundError(random_path)

        cemc = pd.read_csv(
            metrics_path,
            usecols=["trial", "temp_idx", "temperature", "method", "tau", "mse"],
        )
        cemc = cemc.loc[cemc["method"].eq("cemc"), ["trial", "temp_idx", "temperature", "tau", "mse"]]
        cemc_frames.append(cemc)

        random = pd.read_csv(random_path, usecols=["trial", "tau", "mse"])
        random_frames.append(random)

    return pd.concat(cemc_frames, ignore_index=True), pd.concat(random_frames, ignore_index=True)


def _validate_cemc(cemc: pd.DataFrame) -> None:
    trials = set(cemc["trial"].astype(int).unique())
    missing_trials = sorted(_expected_trials() - trials)
    extra_trials = sorted(trials - _expected_trials())
    if missing_trials or extra_trials:
        raise ValueError(f"CEMC trial mismatch: missing={missing_trials[:10]}, extra={extra_trials[:10]}")

    temps = list(cemc.sort_values("temp_idx")["temperature"].drop_duplicates())
    if temps != TEMPERATURES:
        raise ValueError(f"Unexpected CEMC temperatures: {temps}")

    duplicate_keys = cemc.duplicated(["trial", "temperature"]).sum()
    if duplicate_keys:
        raise ValueError(f"CEMC has {duplicate_keys} duplicate trial/temperature rows")

    expected_rows = N_TRIALS * len(TEMPERATURES)
    if len(cemc) != expected_rows:
        raise ValueError(f"CEMC row count mismatch: got {len(cemc)}, expected {expected_rows}")

    counts = cemc.groupby("trial")["temperature"].nunique()
    bad_trials = counts[counts.ne(len(TEMPERATURES))]
    if not bad_trials.empty:
        raise ValueError(f"Trials with incomplete temperature coverage: {bad_trials.head().to_dict()}")


def _validate_random(random: pd.DataFrame, cemc_trials: set[int]) -> None:
    trials = set(random["trial"].astype(int).unique())
    missing_trials = sorted(_expected_trials() - trials)
    extra_trials = sorted(trials - _expected_trials())
    if missing_trials or extra_trials:
        raise ValueError(f"Random trial mismatch: missing={missing_trials[:10]}, extra={extra_trials[:10]}")

    if trials != cemc_trials:
        raise ValueError("Random and CEMC trial sets do not match")

    duplicate_trials = random.duplicated(["trial"]).sum()
    if duplicate_trials:
        raise ValueError(f"Random metrics has {duplicate_trials} duplicate trial rows")

    if len(random) != N_TRIALS:
        raise ValueError(f"Random row count mismatch: got {len(random)}, expected {N_TRIALS}")


def export_csvs(project_dir: Path) -> tuple[Path, Path]:
    results_dir = project_dir / "results"
    cemc, random = _load_shard_tables(results_dir)
    _validate_cemc(cemc)
    _validate_random(random, set(cemc["trial"].astype(int).unique()))

    cemc_long = (
        cemc.sort_values(["trial", "temp_idx"])
        .melt(
            id_vars=["trial", "temperature"],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        .sort_values(
            ["trial", "temperature", "metric"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        [["trial", "temperature", "metric", "value"]]
    )

    # Stable metric order within each trial/temperature: tau first, mse second.
    metric_order = pd.CategoricalDtype(categories=METRICS, ordered=True)
    cemc_long["metric"] = cemc_long["metric"].astype(metric_order)
    cemc_long = cemc_long.sort_values(
        ["trial", "temperature", "metric"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    cemc_long["metric"] = cemc_long["metric"].astype(str)

    expected_long_rows = N_TRIALS * len(TEMPERATURES) * len(METRICS)
    if len(cemc_long) != expected_long_rows:
        raise ValueError(f"CEMC long row count mismatch: got {len(cemc_long)}, expected {expected_long_rows}")

    paired = (
        random.rename(columns={"tau": "random_tau", "mse": "random_mse"})
        .sort_values("trial")
        [["trial", "random_tau", "random_mse"]]
    )

    cemc_out = project_dir / "cemc_temperature_metric_by_trial.csv"
    paired_out = project_dir / "paired_delta_by_trial.csv"
    cemc_long.to_csv(cemc_out, index=False)
    paired.to_csv(paired_out, index=False)
    return cemc_out, paired_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="UQMC project directory containing results/shard_* directories.",
    )
    args = parser.parse_args()

    cemc_out, paired_out = export_csvs(args.project_dir.resolve())
    print(f"Wrote {cemc_out}")
    print(f"Wrote {paired_out}")


if __name__ == "__main__":
    main()
