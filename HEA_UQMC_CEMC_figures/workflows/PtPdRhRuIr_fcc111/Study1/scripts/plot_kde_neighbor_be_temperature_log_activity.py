#!/usr/bin/env python3
"""Plot log-activity KDE-peak-neighbor BE shifts and temperature probability.

The score input is the log-activity-domain result produced by
select_best_score_activity_domain.py --domain log.  The highest-probability
tau-positive KDE point is used as the peak, and the 100 closest trials in the
KDE Mahalanobis metric are selected by default.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")
METRICS = ("tau", "mse", "crps")


def crps_column(frame: pd.DataFrame) -> str:
    for name in ("crps", "crps_scaled"):
        if name in frame:
            return name
    raise ValueError("score CSV has neither crps nor crps_scaled")


def load_points(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    ccol = crps_column(frame)
    frame = frame.rename(columns={ccol: "crps"}) if ccol != "crps" else frame
    for col in ("trial", "temperature", *METRICS):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["trial", "temperature", *METRICS])
    frame["trial"] = frame["trial"].astype(int)
    return frame.drop_duplicates("trial", keep="last").reset_index(drop=True)


def ensure_shifts(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exe = args.output_dir / "export_trial_be_shifts"
    csv = args.output_dir / "trial_be_shifts.csv"
    subprocess.run(["g++", "-O2", "-std=c++11", str(args.shift_exporter), "-o", str(exe)], check=True)
    with csv.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [str(exe), str(args.n_trials), str(args.random_seed), str(args.be_mean), str(args.be_sigma)],
            stdout=handle, check=True,
        )
    return pd.read_csv(csv)


def kde_neighbors(points: pd.DataFrame, n_neighbors: int) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    xyz = points[list(METRICS)].to_numpy(float)
    kde = gaussian_kde(xyz.T)
    density = kde(xyz.T)
    probability = density / density.sum()
    candidate_indices = np.flatnonzero(points["tau"].to_numpy(float) > 0.0)
    if candidate_indices.size == 0:
        raise ValueError("No trial has tau > 0")
    peak_index = candidate_indices[np.argmax(probability[candidate_indices])]
    peak = points.iloc[peak_index]
    delta = xyz - xyz[peak_index]
    inv_cov = np.linalg.pinv(kde.covariance)
    distance2 = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    order = np.argsort(distance2, kind="stable")[:min(n_neighbors, len(points))]
    selected = points.iloc[order].copy()
    selected["kde_density"] = density[order]
    selected["kde_probability"] = probability[order]
    selected["kde_mahalanobis_distance"] = np.sqrt(np.maximum(distance2[order], 0.0))
    selected["neighbor_rank"] = np.arange(1, len(selected) + 1)
    return selected, peak, probability


def plot_3d(points: pd.DataFrame, peak: pd.Series, probability: np.ndarray, output: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points.tau, points.mse, points.crps, c=probability, s=7, alpha=0.22, cmap="viridis")
    ax.scatter(peak.tau, peak.mse, peak.crps, marker="X", s=190, color="red", edgecolors="white", linewidths=1.0,
               label=f"trial {int(peak.trial)} (tau>0 KDE peak)")
    ax.set_xlabel("Kendall tau"); ax.set_ylabel("MSE"); ax.set_zlabel("CRPS")
    ax.set_title(f"Log-activity best-temperature CEMC KDE (P sum={probability.sum():.3f}, N={len(points):,})")
    ax.legend(frameon=False)
    fig.colorbar(sc, ax=ax, shrink=0.66, pad=0.09, label="Normalized KDE probability")
    fig.tight_layout(); fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_summary(neighbors: pd.DataFrame, peak: pd.Series, output: Path, bin_width: float, dpi: int) -> None:
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.0, 1.4), hspace=0.35, wspace=0.25)
    histogram_axes = []
    histogram_max_count = 0
    for index, element in enumerate(ELEMENTS):
        ax = fig.add_subplot(grid[0, index])
        column = f"be_shift_{element}"
        values = neighbors[column].to_numpy(float)
        left = np.floor(values.min() / bin_width) * bin_width
        right = np.ceil(values.max() / bin_width) * bin_width
        if right <= left: right = left + bin_width
        edges = np.arange(left, right + bin_width * 1.01, bin_width)
        counts, _ = np.histogram(values, bins=edges)
        histogram_max_count = max(histogram_max_count, int(counts.max()))
        modal = int(np.argmax(counts))
        colors = ["#5375C6"] * len(counts); colors[modal] = "#D98245"
        ax.bar(edges[:-1], counts, width=bin_width * 0.92, align="edge", color=colors, edgecolor="white", linewidth=0.4)
        in_bin = (values >= edges[modal]) & (values < edges[modal + 1])
        if modal == len(counts) - 1: in_bin |= values == edges[modal + 1]
        representative = neighbors.loc[in_bin].sort_values("neighbor_rank").iloc[0]
        selected_value = float(representative[column])
        ax.axvline(selected_value, color="#C9342B", linestyle="--", linewidth=1.8)
        ax.set_title(f"{element}: {selected_value:.2f} eV\ntrial {int(representative.trial)}; bin count {counts[modal]}/{len(neighbors)}")
        ax.set_xlabel("BE shift (eV)"); ax.grid(axis="y", alpha=0.22)
        if index == 0: ax.set_ylabel(f"Count among {len(neighbors)} trials")
        histogram_axes.append(ax)

    shared_ymax = max(1.0, histogram_max_count * 1.08)
    for histogram_ax in histogram_axes:
        histogram_ax.set_ylim(0.0, shared_ymax)

    ax = fig.add_subplot(grid[1, 1:4])
    temperatures = np.arange(300, 2001, 100)
    probabilities = neighbors.temperature.value_counts().reindex(temperatures, fill_value=0).to_numpy(float) / len(neighbors)
    ax.plot(temperatures, probabilities, color="#B5524E", marker="o", linewidth=1.8, markersize=6)
    ax.set_xlabel("Annealing temperature (K)"); ax.set_ylabel(f"Probability among {len(neighbors)} trials")
    ax.set_xticks(temperatures[::2]); ax.grid(alpha=0.25)
    ax.set_title("Log-activity KDE-neighbor best-temperature probability")
    fig.suptitle(
        f"Log-activity CEMC tau/MSE/CRPS KDE top-{len(neighbors)} BE shifts\n"
        f"peak trial {int(peak.trial)}; fixed {bin_width:.2f} eV histogram; orange = modal bin; red dashed = selected trial value",
        fontsize=15,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--score-input", type=Path, default=Path("results/crps_selection_activity_log/best_temperature_by_trial_crps.csv"))
    p.add_argument("--shift-exporter", type=Path, default=Path("scripts/export_trial_be_shifts_log_activity.cpp"))
    p.add_argument("--output-dir", type=Path, default=Path("results/kde_neighbor_be_temperature_log"))
    p.add_argument("--n-trials", type=int, default=10000)
    p.add_argument("--n-neighbors", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=20260706)
    p.add_argument("--be-mean", type=float, default=0.04233)
    p.add_argument("--be-sigma", type=float, default=0.2604)
    p.add_argument("--bin-width", type=float, default=0.10)
    p.add_argument("--dpi", type=int, default=240)
    args = p.parse_args()

    points = load_points(args.score_input)
    shifts = ensure_shifts(args)
    neighbors, peak, probability = kde_neighbors(points, args.n_neighbors)
    neighbors = neighbors.merge(shifts, on="trial", how="left", validate="one_to_one")
    if neighbors[[f"be_shift_{e}" for e in ELEMENTS]].isna().any().any():
        raise ValueError("Missing reconstructed BE shifts for selected trials")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighbors.to_csv(args.output_dir / "log_activity_kde_peak_nearest_trials.csv", index=False)
    plot_3d(points, peak, probability, args.output_dir / "log_activity_kde_3d_top100.png", args.dpi)
    plot_summary(neighbors, peak, args.output_dir / "log_activity_top100_be_shifts_temperature_probability.png", args.bin_width, args.dpi)
    print(f"KDE peak trial: {int(peak.trial)}")
    print(f"Wrote {args.output_dir / 'log_activity_kde_peak_nearest_trials.csv'}")
    print(f"Wrote {args.output_dir / 'log_activity_kde_3d_top100.png'}")
    print(f"Wrote {args.output_dir / 'log_activity_top100_be_shifts_temperature_probability.png'}")

if __name__ == "__main__":
    main()
