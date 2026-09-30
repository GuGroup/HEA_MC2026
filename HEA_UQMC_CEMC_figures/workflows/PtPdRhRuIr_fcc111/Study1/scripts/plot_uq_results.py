#!/usr/bin/env python3
"""Make quick UQ comparison plots from v1.0 plot_data CSV files.

Usage for one ML:
  python scripts/plot_uq_results.py --plot-data uq_ML1_full/results/plot_data \
      --output uq_ML1_full/results/plots

Usage after combining all ML files with collect_plot_data.py:
  python scripts/plot_uq_results.py --plot-data all_ml_plot_data --output all_ml_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _group_suffix(df: pd.DataFrame, ml_value) -> str:
    if ml_value is None:
        return ""
    return f"_ML{ml_value}"


def plot_tau_mse_scatter(plot_data: Path, outdir: Path) -> None:
    path = plot_data / "tau_mse_scatter_by_trial.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    groups = [(None, df)] if "ml" not in df.columns else list(df.groupby("ml"))
    for ml_value, subdf in groups:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for method, g in subdf.groupby("method"):
            ax.scatter(g["mse"], g["tau"], s=18, alpha=0.65, label=method)
        ax.set_xlabel("MSE after scaling to [-1, 0] (lower is better)")
        ax.set_ylabel("Kendall tau (higher is better)")
        title = "Trial-wise tau/MSE: random vs selected CEMC"
        if ml_value is not None:
            title += f" ML{ml_value}"
        ax.set_title(title)
        ax.legend(frameon=False)
        _save(fig, outdir / f"tau_mse_scatter{_group_suffix(subdf, ml_value)}.png")


def plot_delta_histograms(plot_data: Path, outdir: Path) -> None:
    path = plot_data / "paired_delta_by_trial.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    groups = [(None, df)] if "ml" not in df.columns else list(df.groupby("ml"))
    specs = [
        ("delta_tau_cemc_minus_random", "CEMC tau improvement", "CEMC tau - random tau"),
        ("delta_mse_random_minus_cemc", "CEMC MSE improvement", "random MSE - CEMC MSE"),
        ("delta_score_cemc_minus_random", "CEMC score improvement", "CEMC score - random score"),
    ]
    for ml_value, subdf in groups:
        for col, title, xlabel in specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.hist(subdf[col].dropna(), bins=40, alpha=0.85)
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Number of UQ trials")
            full_title = title if ml_value is None else f"{title} ML{ml_value}"
            ax.set_title(full_title)
            _save(fig, outdir / f"hist_{col}{_group_suffix(subdf, ml_value)}.png")


def _ci_columns(df: pd.DataFrame):
    """Return (low_col, high_col) for 95% CI columns used by v0.5/v1.0 files."""
    if "mean_ci95_low" in df.columns and "mean_ci95_high" in df.columns:
        return "mean_ci95_low", "mean_ci95_high"
    if "ci95_mean_low" in df.columns and "ci95_mean_high" in df.columns:
        return "ci95_mean_low", "ci95_mean_high"
    raise KeyError("Could not find 95% CI columns")


def plot_interval_summary(plot_data: Path, outdir: Path) -> None:
    path = plot_data / "method_metric_interval_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    # v1.0 file columns are: ml,quantity,metric,n_trials,mean,sd,sem,ci95_mean_low,...
    if "group" not in df.columns and "quantity" in df.columns:
        df = df.rename(columns={"quantity": "group"})
    low_col, high_col = _ci_columns(df)
    groups = [(None, df)] if "ml" not in df.columns else list(df.groupby("ml"))
    for ml_value, subdf in groups:
        for metric in ["tau", "mse"]:
            g = subdf[subdf["metric"] == metric].copy().reset_index(drop=True)
            # Keep the direct method distributions only for this plot.
            if "group" in g.columns:
                g = g[g["group"].isin(["cemc_selected", "random"])]
            if g.empty:
                continue
            x = range(len(g))
            lower = g["mean"] - g[low_col]
            upper = g[high_col] - g["mean"]
            fig, ax = plt.subplots(figsize=(5.6, 4.0))
            ax.errorbar(x, g["mean"], yerr=[lower, upper], fmt="o", capsize=4)
            ax.set_xticks(list(x))
            ax.set_xticklabels(g["group"], rotation=20, ha="right")
            ax.set_ylabel(metric)
            direction = "higher is better" if metric == "tau" else "lower is better"
            title = f"Mean {metric} with 95% CI ({direction})"
            if ml_value is not None:
                title += f" ML{ml_value}"
            ax.set_title(title)
            _save(fig, outdir / f"mean_ci_{metric}{_group_suffix(g, ml_value)}.png")

def plot_paired_delta_interval(plot_data: Path, outdir: Path) -> None:
    path = plot_data / "paired_delta_interval_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    low_col, high_col = _ci_columns(df)
    groups = [(None, df)] if "ml" not in df.columns else list(df.groupby("ml"))
    for ml_value, subdf in groups:
        subdf = subdf.reset_index(drop=True)
        if subdf.empty:
            continue
        x = range(len(subdf))
        lower = subdf["mean_delta"] - subdf[low_col]
        upper = subdf[high_col] - subdf["mean_delta"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.errorbar(x, subdf["mean_delta"], yerr=[lower, upper], fmt="o", capsize=4)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(list(x))
        labels = subdf["metric"].astype(str).tolist() if "metric" in subdf.columns else subdf["comparison"].astype(str).tolist()
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Paired mean delta; positive favors CEMC")
        title = "CEMC - random paired improvement with 95% CI"
        if ml_value is not None:
            title += f" ML{ml_value}"
        ax.set_title(title)
        _save(fig, outdir / f"paired_delta_mean_ci{_group_suffix(subdf, ml_value)}.png")

def plot_temperature_summary(plot_data: Path, outdir: Path) -> None:
    summary_path = plot_data / "temperature_metric_summary.csv"
    counts_path = plot_data / "temperature_selection_counts.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        groups = [(None, df)] if "ml" not in df.columns else list(df.groupby("ml"))
        for ml_value, subdf in groups:
            subdf = subdf.sort_values("temperature", ascending=False)
            # v1.0 long format: temperature, metric, mean, ci95_mean_low, ci95_mean_high.
            if "metric" in subdf.columns and "mean" in subdf.columns:
                low_col, high_col = _ci_columns(subdf)
                metrics = [m for m in ["tau", "mse", "score_tau_minus_mse"] if m in set(subdf["metric"])]
                for metric in metrics:
                    g = subdf[subdf["metric"] == metric].copy().sort_values("temperature", ascending=False)
                    fig, ax = plt.subplots(figsize=(7.0, 4.2))
                    ax.plot(g["temperature"], g["mean"], marker="o")
                    ax.fill_between(g["temperature"], g[low_col], g[high_col], alpha=0.2)
                    ax.set_xlabel("CEMC snapshot temperature (K)")
                    ax.set_ylabel(metric)
                    title = f"CEMC {metric} across snapshot temperatures"
                    if ml_value is not None:
                        title += f" ML{ml_value}"
                    ax.set_title(title)
                    ax.invert_xaxis()
                    safe_metric = metric.replace("_tau_minus_mse", "")
                    _save(fig, outdir / f"temperature_{safe_metric}_summary{_group_suffix(g, ml_value)}.png")
            else:
                # Backward-compatible wide format.
                for metric in ["tau", "mse", "score"]:
                    mean_col = f"{metric}_mean"
                    low_col = f"{metric}_mean_ci95_low"
                    high_col = f"{metric}_mean_ci95_high"
                    if mean_col not in subdf.columns:
                        continue
                    fig, ax = plt.subplots(figsize=(7.0, 4.2))
                    ax.plot(subdf["temperature"], subdf[mean_col], marker="o")
                    ax.fill_between(subdf["temperature"], subdf[low_col], subdf[high_col], alpha=0.2)
                    ax.set_xlabel("CEMC snapshot temperature (K)")
                    ax.set_ylabel(metric)
                    title = f"CEMC {metric} across snapshot temperatures"
                    if ml_value is not None:
                        title += f" ML{ml_value}"
                    ax.set_title(title)
                    ax.invert_xaxis()
                    _save(fig, outdir / f"temperature_{metric}_summary{_group_suffix(subdf, ml_value)}.png")

    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        groups = [(None, counts)] if "ml" not in counts.columns else list(counts.groupby("ml"))
        for ml_value, subdf in groups:
            subdf = subdf.sort_values("temperature", ascending=False)
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.bar(subdf["temperature"].astype(str), subdf["fraction_selected"])
            ax.set_xlabel("Selected CEMC temperature (K)")
            ax.set_ylabel("Fraction of UQ trials")
            title = "Temperature selected by max(tau - MSE)"
            if ml_value is not None:
                title += f" ML{ml_value}"
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            _save(fig, outdir / f"temperature_selection_counts{_group_suffix(subdf, ml_value)}.png")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-data", required=True, help="Path to results/plot_data or combined all_ml_plot_data")
    parser.add_argument("--output", default="figures", help="Output directory for PNG files")
    args = parser.parse_args()
    plot_data = Path(args.plot_data).resolve()
    outdir = Path(args.output).resolve()
    if not plot_data.exists():
        raise SystemExit(f"Missing plot_data directory: {plot_data}")
    plot_tau_mse_scatter(plot_data, outdir)
    plot_delta_histograms(plot_data, outdir)
    plot_interval_summary(plot_data, outdir)
    plot_paired_delta_interval(plot_data, outdir)
    plot_temperature_summary(plot_data, outdir)
    print(f"Wrote plots under {outdir}")


if __name__ == "__main__":
    main()
