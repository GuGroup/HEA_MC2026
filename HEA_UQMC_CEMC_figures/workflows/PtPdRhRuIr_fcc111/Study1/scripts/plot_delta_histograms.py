#!/usr/bin/env python3
"""Plot trial-wise CEMC improvement histograms for UQMC results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


PREFERRED_INPUTS = (
    Path("plots/mse_distribution_cemc_vs_random_trial0-999_data.csv"),
    Path("plots/metric_distribution_rows.csv"),
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(
        columns={
            "cemc_selected_temp": "cemc_selected_temperature",
            "cemc_selected_tau": "cemc_tau",
            "cemc_selected_mse": "cemc_mse",
            "cemc_selected_score": "cemc_score",
            "cemc_score_tau_minus_mse": "cemc_score",
            "random_score_tau_minus_mse": "random_score",
        }
    )

    for col in ("trial", "cemc_mse", "random_mse", "cemc_tau", "random_tau"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "delta_mse_random_minus_cemc" not in df.columns:
        if not {"random_mse", "cemc_mse"}.issubset(df.columns):
            raise SystemExit("Missing columns needed for random MSE - CEMC MSE")
        df["delta_mse_random_minus_cemc"] = df["random_mse"] - df["cemc_mse"]

    if "delta_tau_cemc_minus_random" not in df.columns:
        if not {"cemc_tau", "random_tau"}.issubset(df.columns):
            raise SystemExit("Missing columns needed for CEMC tau - random tau")
        df["delta_tau_cemc_minus_random"] = df["cemc_tau"] - df["random_tau"]

    df["delta_mse_random_minus_cemc"] = pd.to_numeric(
        df["delta_mse_random_minus_cemc"], errors="coerce"
    )
    df["delta_tau_cemc_minus_random"] = pd.to_numeric(
        df["delta_tau_cemc_minus_random"], errors="coerce"
    )

    if "trial" in df.columns:
        df = df.dropna(subset=["trial"]).copy()
        df["trial"] = df["trial"].astype(int)
        df = df.drop_duplicates(subset=["trial"], keep="last").sort_values("trial")

    return df


def load_trial_rows(results_root: Path, input_csv: Path | None, world_size: int | None) -> pd.DataFrame:
    if input_csv is not None:
        if not input_csv.exists():
            raise SystemExit(f"Missing input CSV: {input_csv}")
        return normalize_columns(pd.read_csv(input_csv))

    for rel_path in PREFERRED_INPUTS:
        path = results_root / rel_path
        if path.exists() and path.stat().st_size > 0:
            return normalize_columns(pd.read_csv(path))

    if world_size is None:
        shard_dirs = sorted(results_root.glob("shard_*"))
    else:
        shard_dirs = [results_root / f"shard_{idx:02d}" for idx in range(world_size)]

    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / "paired_comparison_by_trial.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["source_shard"] = shard_dir.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No paired comparison CSVs found under {results_root}")

    return normalize_columns(pd.concat(frames, ignore_index=True))


def filter_trials(df: pd.DataFrame, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    if "trial" not in df.columns:
        return df
    out = df
    if trial_min is not None:
        out = out[out["trial"] >= trial_min]
    if trial_max is not None:
        out = out[out["trial"] <= trial_max]
    return out.copy()


def probability_legend_labels(df: pd.DataFrame) -> list[str]:
    cols = ["delta_mse_random_minus_cemc", "delta_tau_cemc_minus_random"]
    clean = df[cols].dropna()
    if clean.empty:
        return [
            "P(DeltaMSE > 0) = n/a",
            "P(DeltaTau > 0) = n/a",
            "P(DeltaMSE > 0, DeltaTau > 0) = n/a",
        ]

    mse_positive = clean["delta_mse_random_minus_cemc"] > 0
    tau_positive = clean["delta_tau_cemc_minus_random"] > 0
    return [
        f"P(DeltaMSE > 0) = {mse_positive.mean():.3f}",
        f"P(DeltaTau > 0) = {tau_positive.mean():.3f}",
        f"P(DeltaMSE > 0, DeltaTau > 0) = {(mse_positive & tau_positive).mean():.3f}",
    ]


def plot_histogram(
    values: pd.Series,
    *,
    output: Path,
    title: str,
    xlabel: str,
    bins: int,
    dpi: int,
    legend_labels: list[str],
) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise SystemExit(f"No finite values for {title}")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.hist(clean, bins=bins, color="#4f81bd", alpha=0.85, edgecolor="#4f81bd", linewidth=0.25)
    ax.axvline(0.0, color="#3f76b5", linestyle="--", linewidth=2.0, alpha=0.9)
    handles = [
        Patch(facecolor="#4f81bd", edgecolor="#4f81bd", alpha=0.85, label="Trial count"),
        Line2D([0], [0], color="#3f76b5", linestyle="--", linewidth=2.0, label="Delta = 0"),
    ]
    handles.extend(Line2D([], [], color="none", label=label) for label in legend_labels)
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=11)
    ax.set_title(title, fontsize=22, pad=12)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel("Number of UQMC trials", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return {
        "n": int(clean.size),
        "n_positive": int((clean > 0).sum()),
        "p_positive": float((clean > 0).mean()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "q025": float(clean.quantile(0.025)),
        "q975": float(clean.quantile(0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing UQMC results")
    parser.add_argument("--input-csv", default="", help="Optional combined paired-comparison CSV")
    parser.add_argument("--world-size", type=int, default=4, help="Number of shard directories to scan")
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400", help="Text used in titles and output filenames")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--summary-csv", default="", help="Optional summary CSV path")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    output_dir = Path(args.output_dir).resolve()
    label = args.label.strip()
    suffix = label.replace(" ", "_") if label else "1400"

    df = load_trial_rows(results_root, input_csv, args.world_size)
    df = filter_trials(df, args.trial_min, args.trial_max)
    if df.empty:
        raise SystemExit("No rows remain after trial filtering")

    mse_output = output_dir / f"hist_delta_mse_random_minus_cemc_{suffix}.png"
    tau_output = output_dir / f"hist_delta_tau_cemc_minus_random_{suffix}.png"
    legend_labels = probability_legend_labels(df)

    summaries = []
    mse_summary = plot_histogram(
        df["delta_mse_random_minus_cemc"],
        output=mse_output,
        title=f"CEMC MSE improvement {label}".rstrip(),
        xlabel="random MSE - CEMC MSE",
        bins=args.bins,
        dpi=args.dpi,
        legend_labels=legend_labels,
    )
    mse_summary.update(
        metric="mse",
        delta="random_mse_minus_cemc_mse",
        positive_meaning="CEMC has lower MSE",
        output=str(mse_output),
    )
    summaries.append(mse_summary)

    tau_summary = plot_histogram(
        df["delta_tau_cemc_minus_random"],
        output=tau_output,
        title=f"CEMC tau improvement {label}".rstrip(),
        xlabel="CEMC tau - random tau",
        bins=args.bins,
        dpi=args.dpi,
        legend_labels=legend_labels,
    )
    tau_summary.update(
        metric="tau",
        delta="cemc_tau_minus_random_tau",
        positive_meaning="CEMC has higher tau",
        output=str(tau_output),
    )
    summaries.append(tau_summary)

    summary = pd.DataFrame(summaries)
    summary = summary[
        [
            "metric",
            "delta",
            "positive_meaning",
            "n",
            "n_positive",
            "p_positive",
            "mean",
            "median",
            "q025",
            "q975",
            "output",
        ]
    ]

    summary_path = Path(args.summary_csv).resolve() if args.summary_csv else output_dir / f"hist_delta_summary_{suffix}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(df)} trial rows")
    print(f"Wrote {mse_output}")
    print(f"Wrote {tau_output}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
