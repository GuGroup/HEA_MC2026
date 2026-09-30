#!/usr/bin/env python3
"""Recompute trial/temperature MSE and CRPS in the project's no-log domain.

Stored prediction means and standard deviations are in log-activity space.  For
the no-log comparison, means are exponentiated and independently scaled to
[-1, 0] for every trial/temperature.  Standard deviations are propagated with
the delta method and scaled by the same span.  Kendall tau is copied from the
existing log-domain score table because exponentiation and min-max scaling are
monotone and therefore leave rank correlation unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf


def gaussian_crps(mean: np.ndarray, sd: np.ndarray, obs: np.ndarray) -> np.ndarray:
    sigma = np.abs(np.asarray(sd, dtype=float))
    mean = np.asarray(mean, dtype=float)
    obs = np.asarray(obs, dtype=float)
    safe = np.where(sigma > 1e-14, sigma, 1.0)
    z = (obs - mean) / safe
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    value = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return np.where(sigma > 1e-14, value, np.abs(obs - mean))


def normalized_distance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, higher in (("tau", True), ("mse", False), ("crps", False)):
        values = out[column].to_numpy(float)
        low, high = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-14:
            distance = np.zeros_like(values)
        elif higher:
            distance = (high - values) / (high - low)
        else:
            distance = (values - low) / (high - low)
        out[f"d_{column}"] = distance
    out["best_score"] = np.sqrt(
        (out.d_tau ** 2 + out.d_mse ** 2 + out.d_crps ** 2) / 3.0
    )
    return out


def score_group(group: pd.DataFrame, experimental: np.ndarray) -> dict[str, float | int]:
    group = group.sort_values("composition_index", kind="stable")
    indices = group.composition_index.to_numpy(int)
    if len(indices) != len(experimental) or not np.array_equal(indices, np.arange(len(experimental))):
        raise ValueError(
            f"trial {int(group.trial.iloc[0])}, temperature {int(group.temperature.iloc[0])}: "
            "composition indices do not match the experimental vector"
        )
    log_mean = group.cemc_pred_activity_mean.to_numpy(float)
    log_sd = np.abs(group.cemc_pred_activity_sd.to_numpy(float))
    linear_mean = np.exp(np.clip(log_mean, -700.0, 700.0))
    span = float(np.nanmax(linear_mean) - np.nanmin(linear_mean))
    if not np.isfinite(span) or span < 1e-30:
        pred = np.zeros_like(linear_mean)
        pred_sd = np.zeros_like(linear_mean)
    else:
        pred = -(linear_mean - np.nanmin(linear_mean)) / span
        pred_sd = linear_mean * log_sd / span
    return {
        "trial": int(group.trial.iloc[0]),
        "temperature": int(group.temperature.iloc[0]),
        "mse": float(np.mean((experimental - pred) ** 2)),
        "crps": float(np.mean(gaussian_crps(pred, pred_sd, experimental))),
        "n_compositions": len(group),
    }


def iter_complete_groups(path: Path, chunksize: int):
    columns = ["trial", "temperature", "composition_index", "cemc_pred_activity_mean", "cemc_pred_activity_sd"]
    carry = pd.DataFrame(columns=columns)
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = pd.concat([carry, chunk], ignore_index=True)
        last_trial = chunk.trial.iloc[-1]
        last_temp = chunk.temperature.iloc[-1]
        is_last = (chunk.trial == last_trial) & (chunk.temperature == last_temp)
        complete = chunk.loc[~is_last]
        carry = chunk.loc[is_last].copy()
        for _, group in complete.groupby(["trial", "temperature"], sort=False):
            yield group
    if not carry.empty:
        yield carry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--experimental-nolog", type=Path, default=Path("orr_matched_activity.json"))
    parser.add_argument("--tau-input", type=Path, default=Path("results/crps_selection/crps_by_trial_temperature.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/crps_selection_nolog"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    with args.experimental_nolog.open(encoding="utf-8") as handle:
        records = json.load(handle)
    experimental = np.asarray([records[str(i)] for i in range(len(records))], dtype=float)

    rows = []
    for shard_index in range(args.world_size):
        path = args.results_root / f"shard_{shard_index:02d}" / "predicted_activity_all_snapshots_by_trial_temperature.csv"
        if not path.exists():
            continue
        print(f"Reading {path}", flush=True)
        for group in iter_complete_groups(path, args.chunksize):
            rows.append(score_group(group, experimental))

    scores = pd.DataFrame(rows)
    tau = pd.read_csv(args.tau_input, usecols=["trial", "temperature", "tau"])
    scores = scores.merge(tau, on=["trial", "temperature"], how="left", validate="one_to_one")
    if scores.tau.isna().any():
        raise ValueError("Missing Kendall tau values after merging the existing score table")
    scores = scores[["trial", "temperature", "crps", "tau", "mse", "n_compositions"]].sort_values(
        ["trial", "temperature"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "crps_by_trial_temperature.csv", index=False)

    selected = []
    for _, group in scores.groupby("trial", sort=True):
        ranked = normalized_distance(group)
        selected.append(
            ranked.sort_values(
                ["best_score", "tau", "mse", "crps"],
                ascending=[True, False, True, True],
            ).head(1)
        )
    best = pd.concat(selected, ignore_index=True)
    best.to_csv(args.output_dir / "best_temperature_by_trial_crps.csv", index=False)
    print(f"Wrote {args.output_dir / 'crps_by_trial_temperature.csv'}")
    print(f"Wrote {args.output_dir / 'best_temperature_by_trial_crps.csv'}")


if __name__ == "__main__":
    main()
