#!/usr/bin/env python3
"""Compute paired CEMC-minus-Random metric improvements for cross-masked scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("tau", "mse", "crps")


def load_scores(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trial", "n_compositions", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")

    columns = ["trial", *METRICS, "n_compositions"]
    if method == "cemc" and "temperature" in frame.columns:
        columns.insert(1, "temperature")
    clean = frame.loc[:, columns].copy()
    for column in columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean.isna().any().any():
        bad = clean.index[clean.isna().any(axis=1)].tolist()[:5]
        raise SystemExit(f"{path} contains non-numeric/missing values near rows {bad}")
    clean["trial"] = clean["trial"].astype(int)
    if clean["trial"].duplicated().any():
        duplicated = clean.loc[clean["trial"].duplicated(), "trial"].tolist()[:5]
        raise SystemExit(f"{path} has duplicate trials: {duplicated}")
    return clean.rename(columns={column: f"{column}_{method}" for column in columns if column != "trial"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-dir", type=Path,
        default=Path("results/crps_selection_activity_log_center_cross_masked"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    score_dir = args.score_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else score_dir
    cemc = load_scores(score_dir / "best_temperature_by_trial_crps.csv", "cemc")
    random = load_scores(score_dir / "random_crps_by_trial.csv", "random")

    cemc_trials = set(cemc["trial"])
    random_trials = set(random["trial"])
    if cemc_trials != random_trials:
        raise SystemExit(
            "CEMC and Random trial sets differ: "
            f"CEMC-only={len(cemc_trials - random_trials)}, "
            f"Random-only={len(random_trials - cemc_trials)}"
        )

    paired = cemc.merge(random, on="trial", how="inner", validate="one_to_one")
    if not np.array_equal(
        paired["n_compositions_cemc"].to_numpy(),
        paired["n_compositions_random"].to_numpy(),
    ):
        raise SystemExit("CEMC and Random n_compositions differ for at least one paired trial")

    paired["delta_tau"] = paired["tau_cemc"] - paired["tau_random"]
    paired["delta_mse"] = paired["mse_random"] - paired["mse_cemc"]
    paired["delta_crps"] = paired["crps_random"] - paired["crps_cemc"]
    improvement_columns = ["delta_tau", "delta_mse", "delta_crps"]
    joint = (paired[improvement_columns] > 0).all(axis=1)

    probability_rows = []
    for column in improvement_columns:
        successes = int((paired[column] > 0).sum())
        probability_rows.append({
            "condition": f"p({column} > 0)",
            "successes": successes,
            "n_paired_trials": len(paired),
            "probability": successes / len(paired),
        })
    probability_rows.append({
        "condition": "p(delta_tau > 0, delta_mse > 0, delta_crps > 0)",
        "successes": int(joint.sum()),
        "n_paired_trials": len(paired),
        "probability": float(joint.mean()),
    })
    probabilities = pd.DataFrame(probability_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "paired_cemc_random_metric_deltas.csv"
    probability_path = output_dir / "cemc_better_probabilities.csv"
    json_path = output_dir / "cemc_better_probabilities.json"
    paired.sort_values("trial").to_csv(delta_path, index=False)
    probabilities.to_csv(probability_path, index=False)
    json_path.write_text(
        json.dumps({row["condition"]: row for row in probability_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Paired trials: {len(paired)}")
    print(f"Cross-masked compositions per trial: {int(paired['n_compositions_cemc'].iloc[0])}")
    for row in probability_rows:
        print(
            f"{row['condition']} = {row['probability']:.6f} "
            f"({row['successes']}/{row['n_paired_trials']})"
        )
    print(f"Wrote {delta_path}")
    print(f"Wrote {probability_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
