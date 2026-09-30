#!/usr/bin/env python3
"""Plot P_tau(T) + P_MSE(T) from experiment-matching CEMC metrics.

This does not compare against the random baseline.

For each UQMC trial:
  - best tau temperature = temperature with maximum tau against experiment
  - best MSE temperature = temperature with minimum MSE against experiment

Then, for each temperature T:
  P_tau(T) = fraction of trials where T is the best tau temperature
  P_MSE(T) = fraction of trials where T is the best MSE temperature
  P_sum(T) = P_tau(T) + P_MSE(T)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

    required = {"trial", "temperature", "tau", "mse"}
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


def summarize_best_temperature_probabilities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_best_tau = df.groupby("trial")["tau"].idxmax()
    idx_best_mse = df.groupby("trial")["mse"].idxmin()
    best_tau = df.loc[idx_best_tau, ["trial", "temperature", "tau", "mse"]].copy()
    best_mse = df.loc[idx_best_mse, ["trial", "temperature", "tau", "mse"]].copy()
    best_tau["criterion"] = "tau"
    best_mse["criterion"] = "mse"
    best_by_trial = pd.concat([best_tau, best_mse], ignore_index=True)

    temperatures = sorted(df["temperature"].unique())
    n_trials = df["trial"].nunique()
    tau_counts = best_tau["temperature"].value_counts().reindex(temperatures, fill_value=0)
    mse_counts = best_mse["temperature"].value_counts().reindex(temperatures, fill_value=0)

    summary = pd.DataFrame(
        {
            "temperature": temperatures,
            "n_trials": n_trials,
            "n_best_tau": tau_counts.to_numpy(),
            "n_best_mse": mse_counts.to_numpy(),
        }
    )
    summary["p_tau_given_temperature"] = summary["n_best_tau"] / n_trials
    summary["p_mse_given_temperature"] = summary["n_best_mse"] / n_trials
    summary["p_tau_plus_p_mse"] = (
        summary["p_tau_given_temperature"] + summary["p_mse_given_temperature"]
    )
    summary["n_tau_plus_n_mse"] = summary["n_best_tau"] + summary["n_best_mse"]
    return summary.sort_values("temperature", ascending=False), best_by_trial.sort_values(["criterion", "trial"])


def plot_probability_sum(summary: pd.DataFrame, label: str, output: Path, dpi: int) -> None:
    plot_df = summary.sort_values("temperature")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(
        plot_df["temperature"].astype(str),
        plot_df["p_tau_plus_p_mse"],
        color="#4f81bd",
        edgecolor="#315f8f",
        linewidth=0.6,
    )
    for bar, value in zip(bars, plot_df["p_tau_plus_p_mse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"P_tau(T) + P_MSE(T) across CEMC temperatures {label}".rstrip(), fontsize=15)
    ax.set_xlabel("CEMC snapshot temperature (K)", fontsize=12)
    ax.set_ylabel("P_tau(T) + P_MSE(T)", fontsize=12)
    ax.set_ylim(0, max(1.0, float(plot_df["p_tau_plus_p_mse"].max()) * 1.18))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_probability_components(summary: pd.DataFrame, label: str, output: Path, dpi: int) -> None:
    x = summary["temperature"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, summary["p_tau_given_temperature"], marker="o", linewidth=1.7, label="P_tau(T)")
    ax.plot(x, summary["p_mse_given_temperature"], marker="o", linewidth=1.7, label="P_MSE(T)")
    ax.plot(
        x,
        summary["p_tau_plus_p_mse"],
        marker="o",
        linewidth=2.3,
        color="#d8842f",
        label="P_tau(T) + P_MSE(T)",
    )
    ax.set_title(f"Best-temperature probability score {label}".rstrip(), fontsize=15)
    ax.set_xlabel("CEMC snapshot temperature (K)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing UQMC results")
    parser.add_argument("--input-csv", default="", help="Optional metrics_by_trial_temperature.csv")
    parser.add_argument("--world-size", type=int, default=4, help="Number of shard directories to scan")
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400", help="Text used in titles and output filenames")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    output_dir = Path(args.output_dir).resolve()
    label = args.label.strip()
    suffix = label.replace(" ", "_") if label else "1400"

    df = load_temperature_metrics(results_root, input_csv, args.world_size)
    df = prepare_metrics(df, args.trial_min, args.trial_max)
    summary, best_by_trial = summarize_best_temperature_probabilities(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"temperature_probability_score_{suffix}.csv"
    best_path = output_dir / f"temperature_probability_best_by_trial_{suffix}.csv"
    bar_path = output_dir / f"temperature_probability_score_bar_{suffix}.png"
    line_path = output_dir / f"temperature_probability_score_components_{suffix}.png"

    summary.to_csv(summary_path, index=False)
    best_by_trial.to_csv(best_path, index=False)
    plot_probability_sum(summary, label, bar_path, args.dpi)
    plot_probability_components(summary, label, line_path, args.dpi)

    best = summary.sort_values("p_tau_plus_p_mse", ascending=False).iloc[0]
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {line_path}")
    print(
        "Best P_tau(T) + P_MSE(T) temperature: "
        f"{int(best['temperature'])} K, score={best['p_tau_plus_p_mse']:.3f}, "
        f"P_tau={best['p_tau_given_temperature']:.3f}, "
        f"P_MSE={best['p_mse_given_temperature']:.3f}"
    )


if __name__ == "__main__":
    main()
