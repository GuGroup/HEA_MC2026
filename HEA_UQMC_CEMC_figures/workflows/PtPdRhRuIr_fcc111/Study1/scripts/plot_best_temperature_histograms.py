#!/usr/bin/env python3
"""Plot best-temperature counts across UQMC trials.

For each trial, this script selects:
  - the CEMC temperature with minimum MSE
  - the CEMC temperature with maximum tau
  - the CEMC temperature with maximum tau - MSE score

It then plots one count-labeled bar chart per selection rule.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BEST_RULES = {
    "mse": {
        "column": "mse",
        "kind": "min",
        "title": "Best CEMC temperature by MSE",
        "ylabel": "Number of trials with minimum MSE",
        "filename": "hist_best_mse_temperature_{suffix}.png",
    },
    "tau": {
        "column": "tau",
        "kind": "max",
        "title": "Best CEMC temperature by tau",
        "ylabel": "Number of trials with maximum tau",
        "filename": "hist_best_tau_temperature_{suffix}.png",
    },
    "score": {
        "column": "score_tau_minus_mse",
        "kind": "max",
        "title": "Best CEMC temperature by tau-MSE score",
        "ylabel": "Number of trials with maximum score",
        "filename": "hist_best_score_temperature_{suffix}.png",
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


def select_best_by_trial(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    spec = BEST_RULES[rule]
    col = spec["column"]
    if spec["kind"] == "min":
        idx = df.groupby("trial")[col].idxmin()
    else:
        idx = df.groupby("trial")[col].idxmax()
    return df.loc[idx].sort_values("trial").copy()


def plot_temperature_counts(
    best_df: pd.DataFrame,
    *,
    rule: str,
    label: str,
    output: Path,
    dpi: int,
) -> pd.Series:
    spec = BEST_RULES[rule]
    temperatures = sorted(best_df["temperature"].dropna().astype(int).unique())
    counts = best_df["temperature"].value_counts().reindex(temperatures, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    bars = ax.bar(
        [str(t) for t in counts.index],
        counts.values,
        color="#4f81bd",
        edgecolor="#315f8f",
        linewidth=0.6,
    )

    max_count = int(counts.max()) if not counts.empty else 0
    offset = max(1, round(max_count * 0.015))
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    title = f"{spec['title']} {label}".rstrip()
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel("Temperature (K)", fontsize=15)
    ax.set_ylabel(spec["ylabel"], fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, max_count + max(8, round(max_count * 0.12)))
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing UQMC results")
    parser.add_argument("--input-csv", default="", help="Optional metrics_by_trial_temperature.csv")
    parser.add_argument("--world-size", type=int, default=4, help="Number of shard directories to scan")
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400", help="Text used in titles and output filenames")
    parser.add_argument("--metrics", nargs="+", choices=sorted(BEST_RULES), default=["mse", "tau", "score"])
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

    summary_frames = []
    best_frames = []
    for rule in args.metrics:
        best = select_best_by_trial(df, rule)
        output = output_dir / BEST_RULES[rule]["filename"].format(suffix=suffix)
        counts = plot_temperature_counts(best, rule=rule, label=label, output=output, dpi=args.dpi)

        summary = counts.rename("n_trials").reset_index()
        summary = summary.rename(columns={"index": "temperature"})
        summary["metric"] = rule
        summary["output"] = str(output)
        summary_frames.append(summary[["metric", "temperature", "n_trials", "output"]])

        cols = ["trial", "temperature", "tau", "mse", "score_tau_minus_mse"]
        keep = [c for c in cols if c in best.columns]
        best_out = best[keep].copy()
        best_out["metric"] = rule
        best_frames.append(best_out[["metric"] + keep])
        print(f"Wrote {output} from {len(best)} trials")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    best_df = pd.concat(best_frames, ignore_index=True)

    summary_path = output_dir / f"best_temperature_counts_{suffix}.csv"
    best_path = output_dir / f"best_temperature_by_trial_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
