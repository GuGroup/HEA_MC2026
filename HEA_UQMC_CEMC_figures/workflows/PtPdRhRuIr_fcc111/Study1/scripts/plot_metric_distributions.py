#!/usr/bin/env python3
"""Plot CEMC-selected vs random trial-wise metric distributions.

This script can be run while the UQMC jobs are still running. It reads completed
trial rows from results/shard_XX/selected_temperature_by_trial.csv and overlays
CEMC-selected and random distributions for MSE, tau, or tau-MSE score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "mse": {
        "cemc": "cemc_mse",
        "random": "random_mse",
        "xlabel": "MSE",
        "title": "CEMC vs random: MSE distribution",
        "subtitle": "lower is better",
    },
    "tau": {
        "cemc": "cemc_tau",
        "random": "random_tau",
        "xlabel": "Kendall tau",
        "title": "CEMC vs random: tau distribution",
        "subtitle": "higher is better",
    },
    "score": {
        "cemc": "cemc_score_tau_minus_mse",
        "random": "random_score_tau_minus_mse",
        "xlabel": "tau - MSE",
        "title": "CEMC vs random: tau-MSE score distribution",
        "subtitle": "higher is better",
    },
}


def load_selected_metrics(results_root: Path, world_size: int | None, input_csv: Path | None) -> pd.DataFrame:
    if input_csv is not None:
        if not input_csv.exists():
            raise SystemExit(f"Missing input CSV: {input_csv}")
        return pd.read_csv(input_csv)

    if world_size is None:
        shard_dirs = sorted(results_root.glob("shard_*"))
    else:
        shard_dirs = [results_root / f"shard_{idx:02d}" for idx in range(world_size)]

    frames = []
    for shard in shard_dirs:
        path = shard / "selected_temperature_by_trial.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["source_shard"] = shard.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"No selected_temperature_by_trial.csv files found under {results_root}")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["trial"], keep="last").sort_values("trial")
    return out


def filter_trials(df: pd.DataFrame, max_trial: int) -> pd.DataFrame:
    if max_trial <= 0:
        return df
    if "trial" not in df.columns:
        raise SystemExit("Cannot apply --max-trial because the input data has no trial column.")
    out = df.copy()
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out = out.dropna(subset=["trial"])
    out = out[out["trial"] < max_trial].copy()
    out["trial"] = out["trial"].astype(int)
    if out.empty:
        raise SystemExit(f"No completed trials with trial < {max_trial} were found.")
    return out.sort_values("trial")


def kde_silverman(values: pd.Series, n_grid: int = 512) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("KDE requires at least two finite values.")
    std = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(std, iqr / 1.349) if iqr > 0.0 else std
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(abs(float(np.mean(x))), 1.0) * 1.0e-6
    bandwidth = 0.9 * scale * (x.size ** (-1.0 / 5.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        bandwidth = max(float(np.ptp(x)), 1.0) * 1.0e-6

    pad = 3.0 * bandwidth
    grid = np.linspace(float(np.min(x)) - pad, float(np.max(x)) + pad, n_grid)
    z = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bandwidth * np.sqrt(2.0 * np.pi))
    peak_idx = int(np.argmax(density))
    return grid, density, float(grid[peak_idx]), float(density[peak_idx])


def plot_distribution(
    df: pd.DataFrame,
    metric: str,
    output: Path,
    bins: int,
    label: str,
    dpi: int,
    show_kde: bool,
) -> None:
    spec = METRICS[metric]
    cemc = pd.to_numeric(df[spec["cemc"]], errors="coerce").dropna()
    random = pd.to_numeric(df[spec["random"]], errors="coerce").dropna()
    if cemc.empty or random.empty:
        raise SystemExit(f"No finite {metric} values were found for both methods.")

    colors = {"cemc": "#7ea3c8", "random": "#efae68"}
    line_colors = {"cemc": "#255f99", "random": "#bd6f16"}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(cemc, bins=bins, density=True, alpha=0.55, label="CEMC selected histogram", color=colors["cemc"])
    ax.hist(random, bins=bins, density=True, alpha=0.55, label="Random histogram", color=colors["random"])

    if show_kde:
        for method, values in [("cemc", cemc), ("random", random)]:
            grid, density, peak_x, peak_y = kde_silverman(values)
            method_label = "CEMC selected" if method == "cemc" else "Random"
            ax.plot(grid, density, color=line_colors[method], linewidth=2.0, label=f"{method_label} KDE")
            ax.scatter(
                [peak_x],
                [peak_y],
                s=48,
                color=line_colors[method],
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
                label=f"{method_label} peak: x={peak_x:.4g}, density={peak_y:.4g}",
            )

    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("Density")
    suffix = f", {label}" if label else ""
    ax.set_title(f"{spec['title']}{suffix}\n{spec['subtitle']}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def write_plot_data(df: pd.DataFrame, output: Path) -> None:
    cols = [
        "trial",
        "source_shard",
        "cemc_selected_temp",
        "cemc_tau",
        "cemc_mse",
        "cemc_score_tau_minus_mse",
        "random_tau",
        "random_mse",
        "random_score_tau_minus_mse",
        "delta_tau_cemc_minus_random",
        "delta_mse_random_minus_cemc",
        "delta_score_cemc_minus_random",
    ]
    keep = [c for c in cols if c in df.columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing shard_XX result directories")
    parser.add_argument("--world-size", type=int, default=4, help="Number of shard directories to scan")
    parser.add_argument("--input-csv", default="", help="Optional selected_temperature_by_trial.csv or combined CSV")
    parser.add_argument("--metric", choices=sorted(METRICS), default="mse")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--max-trial", type=int, default=1000, help="Only plot trials with trial < this value. Use 0 to include all completed trials.")
    parser.add_argument("--no-kde", action="store_true", help="Disable the smoothed Gaussian KDE overlay and peak markers")
    parser.add_argument("--label", default="1400 compositions, trials 0-999", help="Text appended to the plot title")
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--output-csv", default="", help="Optional CSV of the rows used for plotting")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    trial_suffix = f"_trial0-{args.max_trial - 1}" if args.max_trial > 0 else ""
    output = Path(args.output).resolve() if args.output else Path("results/plots") / f"{args.metric}_distribution_cemc_vs_random{trial_suffix}.png"

    df = load_selected_metrics(results_root, args.world_size, input_csv)
    df = filter_trials(df, args.max_trial)
    plot_distribution(df, args.metric, output, args.bins, args.label, args.dpi, not args.no_kde)
    if args.output_csv:
        write_plot_data(df, Path(args.output_csv).resolve())
    print(f"Wrote {output} using {len(df)} completed trials")


if __name__ == "__main__":
    main()
