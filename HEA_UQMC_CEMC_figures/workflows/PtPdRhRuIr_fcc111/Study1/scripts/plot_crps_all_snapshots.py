#!/usr/bin/env python3
"""Compute CRPS diagnostics from all-snapshot compact activity predictions.

CEMC rows come from predicted_activity_all_snapshots_by_trial_temperature.csv.
Random baseline rows come from predicted_activity_selected_by_trial.csv, where
random predictions are temperature-independent.

CRPS is computed for a Gaussian predictive distribution N(mean, sd) against the
experimental activity for each composition. Lower CRPS is better. The primary
criterion is scaled activity CRPS, matching the existing tau/MSE scale.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SQRT2 = math.sqrt(2.0)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / SQRT2))


def gaussian_crps(mean: np.ndarray, sd: np.ndarray, obs: np.ndarray) -> np.ndarray:
    sd = np.asarray(sd, dtype=float)
    mean = np.asarray(mean, dtype=float)
    obs = np.asarray(obs, dtype=float)
    out = np.abs(mean - obs)
    mask = np.isfinite(sd) & (sd > 1e-12)
    if np.any(mask):
        z = (obs[mask] - mean[mask]) / sd[mask]
        out[mask] = sd[mask] * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - INV_SQRT_PI)
    return out


def shard_dirs(results_root: Path, world_size: int | None) -> list[Path]:
    if world_size is None:
        return sorted(results_root.glob("shard_*"))
    return [results_root / f"shard_{idx:02d}" for idx in range(world_size)]


def cemc_paths(results_root: Path, world_size: int | None, input_csv: Path | None) -> list[Path]:
    if input_csv is not None:
        return [input_csv]
    paths = [p / "predicted_activity_all_snapshots_by_trial_temperature.csv" for p in shard_dirs(results_root, world_size)]
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def random_paths(results_root: Path, world_size: int | None, random_input_csv: Path | None) -> list[Path]:
    if random_input_csv is not None:
        return [random_input_csv]
    paths = [p / "predicted_activity_selected_by_trial.csv" for p in shard_dirs(results_root, world_size)]
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def compute_summary(paths: list[Path], trial_min: int | None, trial_max: int | None, chunksize: int) -> pd.DataFrame:
    usecols = [
        "trial", "temperature", "cemc_pred_activity_mean", "cemc_pred_activity_sd",
        "cemc_n_runs", "experimental_activity", "cemc_pred_activity_scaled",
        "cemc_pred_activity_scaled_sd", "experimental_activity_scaled",
    ]
    pieces = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            chunk["trial"] = pd.to_numeric(chunk["trial"], errors="coerce")
            chunk = chunk.dropna(subset=["trial", "temperature"])
            chunk["trial"] = chunk["trial"].astype(int)
            if trial_min is not None:
                chunk = chunk[chunk["trial"] >= trial_min]
            if trial_max is not None:
                chunk = chunk[chunk["trial"] <= trial_max]
            if chunk.empty:
                continue
            raw = gaussian_crps(
                chunk["cemc_pred_activity_mean"].to_numpy(float),
                chunk["cemc_pred_activity_sd"].to_numpy(float),
                chunk["experimental_activity"].to_numpy(float),
            )
            scaled = gaussian_crps(
                chunk["cemc_pred_activity_scaled"].to_numpy(float),
                chunk["cemc_pred_activity_scaled_sd"].to_numpy(float),
                chunk["experimental_activity_scaled"].to_numpy(float),
            )
            chunk = chunk.assign(crps_raw=raw, crps_scaled=scaled)
            pieces.append(
                chunk.groupby(["trial", "temperature"], as_index=False).agg(
                    mean_crps_raw=("crps_raw", "mean"),
                    median_crps_raw=("crps_raw", "median"),
                    mean_crps_scaled=("crps_scaled", "mean"),
                    mean_pred_sd_raw=("cemc_pred_activity_sd", "mean"),
                    min_n_runs=("cemc_n_runs", "min"),
                    n_compositions=("crps_raw", "size"),
                )
            )
    if not pieces:
        raise SystemExit("No CEMC CRPS rows found after filtering")
    partial = pd.concat(pieces, ignore_index=True)
    return partial.groupby(["trial", "temperature"], as_index=False).agg(
        mean_crps_raw=("mean_crps_raw", "mean"),
        median_crps_raw=("median_crps_raw", "mean"),
        mean_crps_scaled=("mean_crps_scaled", "mean"),
        mean_pred_sd_raw=("mean_pred_sd_raw", "mean"),
        min_n_runs=("min_n_runs", "min"),
        n_compositions=("n_compositions", "sum"),
    )


def compute_random_summary(paths: list[Path], trial_min: int | None, trial_max: int | None, chunksize: int) -> pd.DataFrame | None:
    if not paths:
        return None
    usecols = [
        "trial", "random_pred_activity_mean", "random_pred_activity_sd", "random_n_runs",
        "experimental_activity", "random_pred_activity_scaled", "experimental_activity_scaled",
    ]
    pieces = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            chunk["trial"] = pd.to_numeric(chunk["trial"], errors="coerce")
            chunk = chunk.dropna(subset=["trial"])
            chunk["trial"] = chunk["trial"].astype(int)
            if trial_min is not None:
                chunk = chunk[chunk["trial"] >= trial_min]
            if trial_max is not None:
                chunk = chunk[chunk["trial"] <= trial_max]
            if chunk.empty:
                continue
            den = chunk.groupby("trial")["random_pred_activity_mean"].transform(lambda s: float(s.max() - s.min()))
            scaled_sd = np.where(np.abs(den.to_numpy(float)) > 1e-30, chunk["random_pred_activity_sd"].to_numpy(float) / np.abs(den.to_numpy(float)), 0.0)
            raw = gaussian_crps(
                chunk["random_pred_activity_mean"].to_numpy(float),
                chunk["random_pred_activity_sd"].to_numpy(float),
                chunk["experimental_activity"].to_numpy(float),
            )
            scaled = gaussian_crps(
                chunk["random_pred_activity_scaled"].to_numpy(float),
                scaled_sd,
                chunk["experimental_activity_scaled"].to_numpy(float),
            )
            chunk = chunk.assign(crps_raw=raw, crps_scaled=scaled)
            pieces.append(
                chunk.groupby("trial", as_index=False).agg(
                    random_mean_crps_raw=("crps_raw", "mean"),
                    random_median_crps_raw=("crps_raw", "median"),
                    random_mean_crps_scaled=("crps_scaled", "mean"),
                    random_mean_pred_sd_raw=("random_pred_activity_sd", "mean"),
                    random_min_n_runs=("random_n_runs", "min"),
                    n_compositions=("crps_raw", "size"),
                )
            )
    if not pieces:
        return None
    partial = pd.concat(pieces, ignore_index=True)
    return partial.groupby("trial", as_index=False).agg(
        random_mean_crps_raw=("random_mean_crps_raw", "mean"),
        random_median_crps_raw=("random_median_crps_raw", "mean"),
        random_mean_crps_scaled=("random_mean_crps_scaled", "mean"),
        random_mean_pred_sd_raw=("random_mean_pred_sd_raw", "mean"),
        random_min_n_runs=("random_min_n_runs", "min"),
        n_compositions=("n_compositions", "sum"),
    )


def interval_low(s: pd.Series, band: str) -> float:
    s = s.dropna()
    if s.empty:
        return float("nan")
    if band == "quantile":
        return float(s.quantile(0.025))
    if len(s) <= 1:
        return float(s.mean())
    return float(s.mean() - 1.96 * s.std(ddof=1) / math.sqrt(len(s)))


def interval_high(s: pd.Series, band: str) -> float:
    s = s.dropna()
    if s.empty:
        return float("nan")
    if band == "quantile":
        return float(s.quantile(0.975))
    if len(s) <= 1:
        return float(s.mean())
    return float(s.mean() + 1.96 * s.std(ddof=1) / math.sqrt(len(s)))


def random_stats(random_summary: pd.DataFrame | None, band: str) -> dict[str, float] | None:
    if random_summary is None or random_summary.empty:
        return None
    s = random_summary["random_mean_crps_scaled"].dropna()
    if s.empty:
        return None
    return {
        "mean": float(s.mean()),
        "low": interval_low(s, band),
        "high": interval_high(s, band),
        "n_trials": int(s.size),
    }


def plot_best_temperature_hist(best: pd.DataFrame, label: str, output: Path, dpi: int, random_summary: pd.DataFrame | None) -> None:
    counts = best["temperature"].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    bars = ax.bar([str(t) for t in counts.index], counts.values, color="#4f81bd", edgecolor="#315f8f", linewidth=0.6)
    max_count = int(counts.max()) if len(counts) else 0
    offset = max(1, round(max_count * 0.015))
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, f"{int(count)}", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Best CEMC temperature by CRPS {label}".rstrip(), fontsize=20, pad=12)
    ax.set_xlabel("Temperature (K)", fontsize=15)
    ax.set_ylabel("Number of trials with minimum mean scaled CRPS", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, max_count + max(8, round(max_count * 0.12)))
    if random_summary is not None and not random_summary.empty:
        merged = best[["trial", "mean_crps_scaled"]].merge(random_summary[["trial", "random_mean_crps_scaled"]], on="trial", how="inner")
        if not merged.empty:
            p_better = float((merged["mean_crps_scaled"] < merged["random_mean_crps_scaled"]).mean())
            txt = f"Random baseline\nmean CRPS = {merged['random_mean_crps_scaled'].mean():.4f}\nP(best CEMC < random) = {p_better:.3f}"
            ax.text(0.985, 0.965, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.9})
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_temperature_trend(summary: pd.DataFrame, label: str, output: Path, dpi: int, random_summary: pd.DataFrame | None, band: str) -> None:
    trend = summary.groupby("temperature", as_index=False).agg(
        mean_crps=("mean_crps_scaled", "mean"),
        low=("mean_crps_scaled", lambda s: interval_low(s, band)),
        high=("mean_crps_scaled", lambda s: interval_high(s, band)),
    ).sort_values("temperature", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = trend["temperature"].to_numpy()
    ax.plot(x, trend["mean_crps"], marker="o", linewidth=1.9, color="#4f81bd", label="CEMC")
    band_label = "95% CI" if band == "ci95" else "2.5-97.5%"
    ax.fill_between(x, trend["low"], trend["high"], color="#4f81bd", alpha=0.18, linewidth=0, label=f"CEMC {band_label}")
    rs = random_stats(random_summary, band)
    if rs is not None:
        ax.axhline(rs["mean"], color="#666666", linestyle="--", linewidth=1.7, label=f"Random mean ({rs['mean']:.4f})")
        ax.axhspan(rs["low"], rs["high"], color="#777777", alpha=0.12, linewidth=0, label=f"Random {band_label}")
    ax.set_title(f"Mean CRPS across CEMC temperatures {label}".rstrip(), fontsize=15)
    ax.set_xlabel("CEMC snapshot temperature (K)", fontsize=12)
    ax.set_ylabel("Mean CRPS, scaled activity", fontsize=12)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_delta_hist(best: pd.DataFrame, random_summary: pd.DataFrame | None, label: str, output: Path, dpi: int) -> pd.DataFrame | None:
    if random_summary is None or random_summary.empty:
        return None
    merged = best[["trial", "temperature", "mean_crps_scaled"]].merge(
        random_summary[["trial", "random_mean_crps_scaled"]], on="trial", how="inner"
    )
    if merged.empty:
        return None
    merged["delta_crps_random_minus_best_cemc"] = merged["random_mean_crps_scaled"] - merged["mean_crps_scaled"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(merged["delta_crps_random_minus_best_cemc"], bins=40, color="#4f81bd", edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    p = float((merged["delta_crps_random_minus_best_cemc"] > 0).mean())
    ax.set_title(f"Random minus best-CEMC CRPS {label}".rstrip(), fontsize=15)
    ax.set_xlabel("Random CRPS - best CEMC CRPS", fontsize=12)
    ax.set_ylabel("Number of trials", fontsize=12)
    ax.text(0.98, 0.95, f"P(CEMC better) = {p:.3f}\nN = {len(merged)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.9})
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--random-input-csv", default="")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--band", choices=["ci95", "quantile"], default="ci95", help="Uncertainty band for CRPS trend plots")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    random_input_csv = Path(args.random_input_csv).resolve() if args.random_input_csv else None
    paths = cemc_paths(results_root, args.world_size, input_csv)
    if not paths:
        raise SystemExit("No predicted_activity_all_snapshots_by_trial_temperature.csv files found")
    summary = compute_summary(paths, args.trial_min, args.trial_max, args.chunksize)
    random_summary = compute_random_summary(random_paths(results_root, args.world_size, random_input_csv), args.trial_min, args.trial_max, args.chunksize)

    suffix = args.label.strip().replace(" ", "_") if args.label.strip() else "1400"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"crps_by_trial_temperature_{suffix}.csv"
    random_path = output_dir / f"random_crps_by_trial_{suffix}.csv"
    best_path = output_dir / f"best_crps_temperature_by_trial_{suffix}.csv"
    hist_path = output_dir / f"hist_best_crps_temperature_{suffix}.png"
    trend_path = output_dir / f"temperature_trend_crps_{suffix}.png"
    delta_path = output_dir / f"hist_delta_crps_random_minus_best_cemc_{suffix}.png"
    delta_csv = output_dir / f"delta_crps_random_minus_best_cemc_{suffix}.csv"

    summary.to_csv(summary_path, index=False)
    if random_summary is not None:
        random_summary.to_csv(random_path, index=False)
    idx = summary.groupby("trial")["mean_crps_scaled"].idxmin()
    best = summary.loc[idx].sort_values("trial")
    best.to_csv(best_path, index=False)
    plot_best_temperature_hist(best, args.label.strip(), hist_path, args.dpi, random_summary)
    plot_temperature_trend(summary, args.label.strip(), trend_path, args.dpi, random_summary, args.band)
    delta = plot_delta_hist(best, random_summary, args.label.strip(), delta_path, args.dpi)
    if delta is not None:
        delta.to_csv(delta_csv, index=False)

    top = best["temperature"].astype(int).value_counts().sort_values(ascending=False).head(1)
    print(f"Wrote {summary_path}")
    if random_summary is not None:
        print(f"Wrote {random_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {trend_path}")
    if delta is not None:
        print(f"Wrote {delta_path}")
        print(f"Wrote {delta_csv}")
    print(f"Most frequent best-CRPS temperature: {int(top.index[0])} K ({int(top.iloc[0])} trials)")
    rs = random_stats(random_summary, args.band)
    if rs is not None:
        print(f"Random baseline mean scaled CRPS: {rs['mean']:.6f} from {rs['n_trials']} trials")


if __name__ == "__main__":
    main()
