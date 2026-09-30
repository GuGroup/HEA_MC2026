#!/usr/bin/env python3
"""Plot method-specific tau/MSE hexbin diagnostics for UQMC 1400 results.

This script reads completed shard outputs, optionally limits the trial range,
selects MC/CEMC-best and Random-best trials by normalized upper-left distance,
and marks both those best trials and 2D Gaussian KDE peak locations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BestFor = Literal["cemc", "random", "either"]


def read_sharded_csv(results_root: Path, filename: str, world_size: int | None) -> pd.DataFrame:
    if world_size is None:
        shard_dirs = sorted(results_root.glob("shard_*"))
    else:
        shard_dirs = [results_root / f"shard_{idx:02d}" for idx in range(world_size)]
    frames = []
    for shard in shard_dirs:
        path = shard / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_shard"] = shard.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"No {filename} files found under {results_root}")
    return pd.concat(frames, ignore_index=True)


def load_selected_metrics(results_root: Path, world_size: int | None, input_csv: Path | None) -> pd.DataFrame:
    if input_csv is not None:
        if not input_csv.exists():
            raise SystemExit(f"Missing input CSV: {input_csv}")
        df = pd.read_csv(input_csv)
    else:
        df = read_sharded_csv(results_root, "selected_temperature_by_trial.csv", world_size)
    df = df.drop_duplicates(subset=["trial"], keep="last")
    numeric_cols = [
        "trial",
        "cemc_selected_temp",
        "cemc_score_tau_minus_mse",
        "cemc_tau",
        "cemc_mse",
        "random_score_tau_minus_mse",
        "random_tau",
        "random_mse",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trial", "cemc_tau", "cemc_mse", "random_tau", "random_mse"])
    df["trial"] = df["trial"].astype(int)
    return df.sort_values("trial").reset_index(drop=True)


def filter_metrics_by_trial(metrics: pd.DataFrame, max_trial: int) -> pd.DataFrame:
    if max_trial <= 0:
        return metrics
    out = metrics[metrics["trial"] < max_trial].copy()
    if out.empty:
        raise SystemExit(f"No completed trials with trial < {max_trial} were found.")
    return out.reset_index(drop=True)


def points_for_best(metrics: pd.DataFrame, best_for: BestFor) -> pd.DataFrame:
    rows = []
    if best_for in {"cemc", "either"}:
        rows.append(pd.DataFrame({
            "trial": metrics["trial"].astype(int),
            "method": "cemc",
            "tau": metrics["cemc_tau"].astype(float),
            "mse": metrics["cemc_mse"].astype(float),
            "score_tau_minus_mse": metrics["cemc_score_tau_minus_mse"].astype(float),
            "selected_temperature": metrics["cemc_selected_temp"].astype(float),
            "source_shard": metrics.get("source_shard", pd.Series([""] * len(metrics))),
        }))
    if best_for in {"random", "either"}:
        rows.append(pd.DataFrame({
            "trial": metrics["trial"].astype(int),
            "method": "random",
            "tau": metrics["random_tau"].astype(float),
            "mse": metrics["random_mse"].astype(float),
            "score_tau_minus_mse": metrics["random_score_tau_minus_mse"].astype(float),
            "selected_temperature": np.nan,
            "source_shard": metrics.get("source_shard", pd.Series([""] * len(metrics))),
        }))
    pts = pd.concat(rows, ignore_index=True)
    pts = pts.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau", "mse"])
    if pts.empty:
        raise SystemExit("No finite tau/MSE points were found.")
    return pts


def select_upper_left(points: pd.DataFrame) -> pd.Series:
    mse = points["mse"].astype(float)
    tau = points["tau"].astype(float)
    mse_range = max(float(mse.max() - mse.min()), 1.0e-15)
    tau_range = max(float(tau.max() - tau.min()), 1.0e-15)
    work = points.copy()
    work["upper_left_distance"] = np.sqrt(
        ((mse - float(mse.min())) / mse_range) ** 2 +
        ((float(tau.max()) - tau) / tau_range) ** 2
    )
    return work.sort_values(
        ["upper_left_distance", "mse", "tau"],
        ascending=[True, True, False],
    ).iloc[0]


def select_best_trials(points: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for method in ["cemc", "random"]:
        method_points = points[points["method"] == method]
        if method_points.empty:
            continue
        selected.append(select_upper_left(method_points))
    if not selected:
        raise SystemExit("No method-specific best trials could be selected.")
    return pd.DataFrame(selected).reset_index(drop=True)


def kde2d_peak(x_values: pd.Series, y_values: pd.Series, n_grid: int = 120) -> tuple[float, float, float]:
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        raise ValueError("2D KDE requires at least two finite points.")

    def bandwidth(vals: np.ndarray) -> float:
        std = float(np.std(vals, ddof=1))
        iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
        scale = min(std, iqr / 1.349) if iqr > 0.0 else std
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(abs(float(np.mean(vals))), 1.0) * 1.0e-6
        bw = 0.9 * scale * (vals.size ** (-1.0 / 6.0))
        if not np.isfinite(bw) or bw <= 0.0:
            bw = max(float(np.ptp(vals)), 1.0) * 1.0e-6
        return bw

    bw_x = bandwidth(x)
    bw_y = bandwidth(y)
    x_grid = np.linspace(float(np.min(x)) - 3.0 * bw_x, float(np.max(x)) + 3.0 * bw_x, n_grid)
    y_grid = np.linspace(float(np.min(y)) - 3.0 * bw_y, float(np.max(y)) + 3.0 * bw_y, n_grid)
    best_density = -np.inf
    best_x = float(x_grid[0])
    best_y = float(y_grid[0])
    norm = x.size * bw_x * bw_y * 2.0 * np.pi
    zx_all = ((x_grid[:, None] - x[None, :]) / bw_x) ** 2
    for y0 in y_grid:
        zy = ((y0 - y) / bw_y) ** 2
        density = np.exp(-0.5 * (zx_all + zy[None, :])).sum(axis=1) / norm
        idx = int(np.argmax(density))
        if float(density[idx]) > best_density:
            best_density = float(density[idx])
            best_x = float(x_grid[idx])
            best_y = float(y0)
    return best_x, best_y, best_density


def plot_hexbin_tau_mse(
    points: pd.DataFrame,
    selected_by_method: pd.DataFrame,
    output: Path,
    gridsize: int,
    show_kde: bool,
) -> None:
    labels = {"cemc": "MC/CEMC selected", "random": "Random"}
    cmaps = {"cemc": "Blues", "random": "Oranges"}
    marker_colors = {"cemc": "#1f4e79", "random": "#9a4f00"}
    selected_lookup = {str(row["method"]): row for _, row in selected_by_method.iterrows()}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=180, constrained_layout=True, sharex=True, sharey=True)
    for ax, method in zip(axes, ["cemc", "random"]):
        grp = points[points["method"] == method].copy()
        if grp.empty:
            ax.set_visible(False)
            continue
        hb = ax.hexbin(grp["mse"], grp["tau"], gridsize=gridsize, mincnt=1, cmap=cmaps[method], linewidths=0.2)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Trial count", rotation=270, labelpad=12)

        selected = selected_lookup.get(method)
        if selected is not None:
            ax.scatter(
                [selected["mse"]], [selected["tau"]], s=150, marker="*",
                color=marker_colors[method], edgecolor="black", linewidth=0.8, zorder=4,
                label=f"Best trial {int(selected['trial'])}: MSE={float(selected['mse']):.4g}, tau={float(selected['tau']):.4g}",
            )
        if show_kde:
            try:
                peak_mse, peak_tau, peak_density = kde2d_peak(grp["mse"], grp["tau"])
                ax.scatter(
                    [peak_mse], [peak_tau], s=62, marker="o", facecolor="white",
                    edgecolor="black", linewidth=1.2, zorder=5,
                    label=f"KDE peak: MSE={peak_mse:.4g}, tau={peak_tau:.4g}, dens={peak_density:.4g}",
                )
            except ValueError:
                pass
        ax.set_title(f"{labels[method]} tau/MSE hexbin")
        ax.set_xlabel("MSE")
        ax.set_ylabel("Kendall tau")
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"Trial-wise tau/MSE density, trials={points['trial'].nunique()}", weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing shard_XX results")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--input-csv", default="", help="Optional selected_temperature_by_trial.csv or combined CSV")
    parser.add_argument("--best-for", choices=["cemc", "random", "either"], default="either")
    parser.add_argument("--max-trial", type=int, default=1000, help="Only use trials with trial < this value. Use 0 for all completed trials.")
    parser.add_argument("--hexbin-gridsize", type=int, default=20)
    parser.add_argument("--no-kde", action="store_true", help="Disable 2D KDE peak markers")
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--selected-csv", default="", help="Optional CSV path for selected best trials")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    trial_suffix = f"_trial0-{args.max_trial - 1}" if args.max_trial > 0 else ""
    output = Path(args.output).resolve() if args.output else Path("results/plots") / f"tau_mse_hexbin_upper_left_selection{trial_suffix}.png"
    selected_csv = Path(args.selected_csv).resolve() if args.selected_csv else output.with_name(output.stem + "_selected_trials.csv")

    metrics = filter_metrics_by_trial(load_selected_metrics(results_root, args.world_size, input_csv), args.max_trial)
    points = points_for_best(metrics, args.best_for)
    selected_by_method = select_best_trials(points)
    plot_hexbin_tau_mse(points, selected_by_method, output, args.hexbin_gridsize, not args.no_kde)
    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_by_method.to_csv(selected_csv, index=False)

    for _, selected in selected_by_method.iterrows():
        print(f"selected method={selected['method']} trial={int(selected['trial'])} tau={float(selected['tau']):.6g} mse={float(selected['mse']):.6g}")
    print(f"wrote {output}")
    print(f"wrote {selected_csv}")


if __name__ == "__main__":
    main()
