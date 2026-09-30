
#!/usr/bin/env python3
"""Select temperatures/trials by equal-weight normalized tau, MSE, and CRPS."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import kendalltau


def read_shards(root: Path, filename: str, world_size: int | None) -> pd.DataFrame:
    shards = [root / f"shard_{idx:02d}" for idx in range(world_size)] if world_size is not None else sorted(root.glob("shard_*"))
    frames = []
    for shard in shards:
        path = shard / filename
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if not df.empty:
                df["source_shard"] = shard.name
                frames.append(df)
    if not frames:
        raise SystemExit(f"No {filename} files found under {root}")
    return pd.concat(frames, ignore_index=True)


def gaussian_crps(mean, sd, obs):
    mean, sd, obs = map(lambda x: np.asarray(x, dtype=float), (mean, sd, obs))
    sigma = np.abs(sd)
    safe = np.where(sigma > 1e-14, sigma, 1.0)
    z = (obs - mean) / safe
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return np.where(sigma > 1e-14, crps, np.abs(obs - mean))


def metric_columns(activity_scale: str) -> dict[str, str]:
    if activity_scale == "scaled":
        return {
            "exp_all": "experimental_activity_scaled",
            "cemc_all_mean": "cemc_pred_activity_scaled",
            "cemc_all_sd": "cemc_pred_activity_scaled_sd",
            "exp_selected": "experimental_activity_scaled",
            "cemc_selected_mean": "cemc_pred_activity_scaled",
            "cemc_selected_sd": "cemc_pred_activity_scaled_sd",
            "random_selected_mean": "random_pred_activity_scaled",
            "random_selected_sd": "random_pred_activity_sd",
            "random_selected_span_mean": "random_pred_activity_mean",
            "random_selected_exp": "experimental_activity_scaled",
            "crps_column": "crps",
        }
    if activity_scale == "raw":
        return {
            "exp_all": "experimental_activity",
            "cemc_all_mean": "cemc_pred_activity_mean",
            "cemc_all_sd": "cemc_pred_activity_sd",
            "exp_selected": "experimental_activity",
            "cemc_selected_mean": "cemc_pred_activity_mean",
            "cemc_selected_sd": "cemc_pred_activity_sd",
            "random_selected_mean": "random_pred_activity_mean",
            "random_selected_sd": "random_pred_activity_sd",
            "random_selected_span_mean": "random_pred_activity_mean",
            "random_selected_exp": "experimental_activity",
            "crps_column": "crps",
        }
    raise SystemExit(f"Unknown activity scale: {activity_scale}")


def kendall_mse(exp: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tau = float(kendalltau(exp, pred, variant="b").statistic)
    mse = float(np.mean((exp - pred) ** 2)) if len(exp) else float("nan")
    return tau, mse


def normalized_distance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, higher in (("tau", True), ("mse", False), ("crps", False)):
        values = out[column].to_numpy(float)
        low, high = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-14:
            distance = np.zeros_like(values, dtype=float)
        elif higher:
            distance = (high - values) / (high - low)
        else:
            distance = (values - low) / (high - low)
        out[f"d_{column}"] = distance
    out["best_score"] = np.sqrt((out["d_tau"] ** 2 + out["d_mse"] ** 2 + out["d_crps"] ** 2) / 3.0)
    return out


def choose_one(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["best_score", "tau", "mse", "crps"], ascending=[True, False, True, True]).head(1)


def build_cemc_candidates(root: Path, world_size: int | None, cols: dict[str, str]) -> pd.DataFrame:
    activity = read_shards(root, "predicted_activity_all_snapshots_by_trial_temperature.csv", world_size)
    required = {"trial", "temperature", "composition_index", cols["exp_all"], cols["cemc_all_mean"], cols["cemc_all_sd"]}
    missing = sorted(required - set(activity.columns))
    if missing:
        raise SystemExit(f"CEMC activity input is missing: {', '.join(missing)}")
    activity = activity[list(required) + [c for c in ["source_shard"] if c in activity.columns]].copy()
    for col in ["trial", "temperature", "composition_index", cols["exp_all"], cols["cemc_all_mean"], cols["cemc_all_sd"]]:
        activity[col] = pd.to_numeric(activity[col], errors="coerce")
    activity = activity.dropna(subset=["trial", "temperature", "composition_index", cols["exp_all"], cols["cemc_all_mean"], cols["cemc_all_sd"]])
    activity["trial"] = activity["trial"].astype(int)
    activity["temperature"] = activity["temperature"].astype(int)
    rows = []
    for (trial, temperature), group in activity.groupby(["trial", "temperature"], sort=True):
        exp = group[cols["exp_all"]].to_numpy(float)
        pred = group[cols["cemc_all_mean"]].to_numpy(float)
        sd = group[cols["cemc_all_sd"]].to_numpy(float)
        tau, mse = kendall_mse(exp, pred)
        crps = float(np.mean(gaussian_crps(pred, sd, exp)))
        rows.append({
            "trial": int(trial),
            "temperature": int(temperature),
            "tau": tau,
            "mse": mse,
            "crps": crps,
            "activity_scale": activity_scale,
            "n_compositions": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values(["trial", "temperature"]).reset_index(drop=True)


def build_random_candidates(root: Path, world_size: int | None, cols: dict[str, str]) -> pd.DataFrame:
    selected = read_shards(root, "predicted_activity_selected_by_trial.csv", world_size)
    required = {"trial", "composition_index", cols["exp_selected"], cols["random_selected_mean"], cols["random_selected_sd"]}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise SystemExit(f"Random activity input is missing: {', '.join(missing)}")
    selected = selected.drop_duplicates(["trial", "composition_index"]).copy()
    for col in ["trial", "composition_index", cols["exp_selected"], cols["random_selected_mean"], cols["random_selected_sd"]]:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["trial", "composition_index", cols["exp_selected"], cols["random_selected_mean"], cols["random_selected_sd"]])
    selected["trial"] = selected["trial"].astype(int)
    random_rows = []
    for trial, group in selected.groupby("trial", sort=True):
        exp = group[cols["exp_selected"]].to_numpy(float)
        pred = group[cols["random_selected_mean"]].to_numpy(float)
        sd = group[cols["random_selected_sd"]].to_numpy(float)
        tau, mse = kendall_mse(exp, pred)
        crps = float(np.mean(gaussian_crps(pred, sd, exp)))
        random_rows.append({
            "trial": int(trial),
            "tau": tau,
            "mse": mse,
            "crps": crps,
            "activity_scale": activity_scale,
            "n_compositions": int(len(group)),
        })
    return pd.DataFrame(random_rows).sort_values("trial").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--activity-scale", choices=["scaled", "raw"], default="scaled")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    global activity_scale
    activity_scale = args.activity_scale
    root = Path(args.results_root)
    outdir = Path(args.output_dir) if args.output_dir else root / ("crps_selection" if args.activity_scale == "scaled" else "crps_selection_raw")
    outdir.mkdir(parents=True, exist_ok=True)
    cols = metric_columns(args.activity_scale)

    cemc = build_cemc_candidates(root, args.world_size, cols)
    cemc.to_csv(outdir / "crps_by_trial_temperature.csv", index=False)

    best_temperatures = []
    for _, group in cemc.groupby("trial", sort=True):
        best_temperatures.append(choose_one(normalized_distance(group)))
    pd.concat(best_temperatures, ignore_index=True).to_csv(outdir / "best_temperature_by_trial_crps.csv", index=False)

    cemc_global = normalized_distance(cemc)
    cemc_global.to_csv(outdir / "cemc_global_candidates.csv", index=False)
    choose_one(cemc_global).to_csv(outdir / "best_trial_cemc_crps.csv", index=False)

    random = build_random_candidates(root, args.world_size, cols)
    random.to_csv(outdir / "random_crps_by_trial.csv", index=False)
    random_global = normalized_distance(random)
    random_global.to_csv(outdir / "random_global_candidates.csv", index=False)
    choose_one(random_global).to_csv(outdir / "best_trial_random_crps.csv", index=False)


if __name__ == "__main__":
    main()
