#!/usr/bin/env python3
"""Plot CEMC metric trends and trial-wise optimum probabilities by temperature."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "mse": {
        "column": "mse",
        "title": "CEMC MSE across snapshot temperatures",
        "ylabel": "MSE",
        "filename": "temperature_trend_mse_{suffix}.png",
    },
    "tau": {
        "column": "tau",
        "title": "CEMC tau across snapshot temperatures",
        "ylabel": "tau",
        "filename": "temperature_trend_tau_{suffix}.png",
    },
    "score": {
        "column": "score_tau_minus_mse",
        "title": "CEMC score_tau_minus_mse across snapshot temperatures",
        "ylabel": "score_tau_minus_mse",
        "filename": "temperature_trend_score_tau_minus_mse_{suffix}.png",
    },
}


def load_temperature_metrics(results_root: Path, input_csv: Path | None, world_size: int | None) -> pd.DataFrame:
    if input_csv is not None:
        if not input_csv.exists():
            raise SystemExit(f"Missing input CSV: {input_csv}")
        return pd.read_csv(input_csv)

    combined = results_root / "combined" / "metrics_by_trial_temperature.csv"
    if combined.exists() and combined.stat().st_size > 0:
        return pd.read_csv(combined)

    if world_size is None:
        shard_dirs = sorted(results_root.glob("shard_*"))
    else:
        shard_dirs = [results_root / f"shard_{idx:02d}" for idx in range(world_size)]

    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / "metrics_by_trial_temperature.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["source_shard"] = shard_dir.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No metrics_by_trial_temperature.csv files found under {results_root}")

    return pd.concat(frames, ignore_index=True)


def prepare_metrics(df: pd.DataFrame, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df = df[df["method"].astype(str).str.lower() == "cemc"]

    required = {"trial", "temperature", "mse", "tau", "score_tau_minus_mse"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    df["trial"] = df["trial"].astype(int)
    df["temperature"] = df["temperature"].astype(int)

    if trial_min is not None:
        df = df[df["trial"] >= trial_min]
    if trial_max is not None:
        df = df[df["trial"] <= trial_max]

    if df.empty:
        raise SystemExit("No rows remain after filtering")

    return df


def load_crps_metrics(path: Path, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Missing CRPS summary: {path}\n"
            "Run scripts/plot_crps_all_snapshots.py first, or pass --crps-input-csv."
        )
    df = pd.read_csv(path)
    required = {"trial", "temperature", "mean_crps_scaled"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"CRPS input is missing required columns: {', '.join(missing)}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    df["trial"] = df["trial"].astype(int)
    df["temperature"] = df["temperature"].astype(int)
    if trial_min is not None:
        df = df[df["trial"] >= trial_min]
    if trial_max is not None:
        df = df[df["trial"] <= trial_max]
    if df.empty:
        raise SystemExit("No CRPS rows remain after filtering")
    return df


def summarize_optimum_probabilities(
    metrics: pd.DataFrame, crps: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_trials = sorted(set(metrics["trial"]) & set(crps["trial"]))
    if not common_trials:
        raise SystemExit("Metric and CRPS inputs have no trials in common")
    metrics = metrics[metrics["trial"].isin(common_trials)].copy()
    crps = crps[crps["trial"].isin(common_trials)].copy()

    best_tau = metrics.loc[
        metrics.groupby("trial")["tau"].idxmax(), ["trial", "temperature", "tau"]
    ].rename(columns={"temperature": "best_tau_temperature", "tau": "best_tau"})
    best_mse = metrics.loc[
        metrics.groupby("trial")["mse"].idxmin(), ["trial", "temperature", "mse"]
    ].rename(columns={"temperature": "best_mse_temperature", "mse": "best_mse"})
    best_crps = crps.loc[
        crps.groupby("trial")["mean_crps_scaled"].idxmin(),
        ["trial", "temperature", "mean_crps_scaled"],
    ].rename(
        columns={
            "temperature": "best_crps_temperature",
            "mean_crps_scaled": "best_mean_crps_scaled",
        }
    )
    best = best_tau.merge(best_mse, on="trial").merge(best_crps, on="trial")

    temperatures = sorted(set(metrics["temperature"]) | set(crps["temperature"]), reverse=True)
    n_trials = len(best)
    summary = pd.DataFrame({"temperature": temperatures})
    summary["n_trials"] = n_trials
    for criterion in ("tau", "mse", "crps"):
        counts = best[f"best_{criterion}_temperature"].value_counts()
        summary[f"n_best_{criterion}"] = summary["temperature"].map(counts).fillna(0).astype(int)
        summary[f"p_best_{criterion}"] = summary[f"n_best_{criterion}"] / n_trials
    return summary, best.sort_values("trial")


def plot_optimum_probabilities(
    summary: pd.DataFrame, label: str, output: Path, dpi: int
) -> None:
    x = summary["temperature"].to_numpy()
    n_trials = int(summary["n_trials"].iloc[0])
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.plot(x, summary["p_best_tau"], marker="o", linewidth=2.0, markersize=6,
            label=r"$P(\tau_{\max}\mid T)$")
    ax.plot(x, summary["p_best_mse"], marker="o", linewidth=2.0, markersize=6,
            label=r"$P(\mathrm{MSE}_{\min}\mid T)$")
    ax.plot(x, summary["p_best_crps"], marker="o", linewidth=2.0, markersize=6,
            label=r"$P(\mathrm{CRPS}_{\min}\mid T)$")
    title_prefix = f"{label}: " if label else ""
    ax.set_title(
        f"{title_prefix}optimum-metric probability vs temperature  (n={n_trials:,})",
        fontsize=17,
        pad=10,
    )
    ax.set_xlabel("Temperature, T (K)", fontsize=13)
    ax.set_ylabel("Probability of being the trial-wise optimum", fontsize=13)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def summarize_by_temperature(df: pd.DataFrame, metric: str, band: str) -> pd.DataFrame:
    col = METRICS[metric]["column"]
    grouped = df.groupby("temperature")[col]

    summary = grouped.agg(mean="mean", median="median", sd="std", n="count").reset_index()
    if band == "quantile":
        q = grouped.quantile([0.025, 0.975]).unstack()
        q = q.rename(columns={0.025: "low", 0.975: "high"}).reset_index()
        summary = summary.merge(q, on="temperature")
    elif band == "sem95":
        sem = summary["sd"] / summary["n"].pow(0.5)
        summary["low"] = summary["mean"] - 1.96 * sem
        summary["high"] = summary["mean"] + 1.96 * sem
    elif band == "sd":
        summary["low"] = summary["mean"] - summary["sd"]
        summary["high"] = summary["mean"] + summary["sd"]
    else:
        raise SystemExit(f"Unknown band: {band}")

    return summary.sort_values("temperature", ascending=False)


def plot_metric(summary: pd.DataFrame, metric: str, label: str, output: Path, dpi: int) -> None:
    spec = METRICS[metric]
    x = summary["temperature"].to_numpy()
    mean = summary["mean"].to_numpy()
    low = summary["low"].to_numpy()
    high = summary["high"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.fill_between(x, low, high, color="#4f81bd", alpha=0.22, linewidth=0)
    ax.plot(x, mean, color="#3f76b5", marker="o", linewidth=1.8, markersize=5)
    ax.set_title(f"{spec['title']} {label}".rstrip(), fontsize=16, pad=8)
    ax.set_xlabel("CEMC snapshot temperature (K)", fontsize=12)
    ax.set_ylabel(spec["ylabel"], fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.invert_xaxis()
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing UQMC results")
    parser.add_argument("--input-csv", default="", help="Optional metrics_by_trial_temperature.csv")
    parser.add_argument(
        "--crps-input-csv", default="",
        help="Optional crps_by_trial_temperature CSV; default is OUTPUT_DIR/crps_by_trial_temperature_SUFFIX.csv",
    )
    parser.add_argument("--world-size", type=int, default=4, help="Number of shard directories to scan")
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400", help="Text used in titles and output filenames")
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=["mse", "tau", "score"])
    parser.add_argument("--band", choices=["quantile", "sem95", "sd"], default="quantile")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    output_dir = Path(args.output_dir).resolve()
    label = args.label.strip()
    suffix = label.replace(" ", "_") if label else "1400"
    crps_input_csv = (
        Path(args.crps_input_csv).resolve()
        if args.crps_input_csv
        else output_dir / f"crps_by_trial_temperature_{suffix}.csv"
    )

    df = load_temperature_metrics(results_root, input_csv, args.world_size)
    df = prepare_metrics(df, args.trial_min, args.trial_max)
    crps_df = load_crps_metrics(crps_input_csv, args.trial_min, args.trial_max)

    summaries = []
    for metric in args.metrics:
        summary = summarize_by_temperature(df, metric, args.band)
        summary["metric"] = metric
        summaries.append(summary[["metric", "temperature", "n", "mean", "median", "sd", "low", "high"]])

        output = output_dir / METRICS[metric]["filename"].format(suffix=suffix)
        plot_metric(summary, metric, label, output, args.dpi)
        print(f"Wrote {output}")

    summary_path = output_dir / f"temperature_metric_trend_summary_{suffix}.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    probability_summary, best_by_trial = summarize_optimum_probabilities(df, crps_df)
    probability_path = output_dir / f"temperature_optimum_metric_probability_{suffix}.csv"
    best_path = output_dir / f"temperature_optimum_metric_best_by_trial_{suffix}.csv"
    probability_plot = output_dir / f"temperature_optimum_metric_probability_{suffix}.png"
    probability_summary.to_csv(probability_path, index=False)
    best_by_trial.to_csv(best_path, index=False)
    plot_optimum_probabilities(probability_summary, label, probability_plot, args.dpi)
    print(f"Wrote {probability_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {probability_plot}")


if __name__ == "__main__":
    main()
