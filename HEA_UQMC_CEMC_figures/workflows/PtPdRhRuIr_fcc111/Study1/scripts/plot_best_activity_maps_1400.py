#!/usr/bin/env python3
"""Plot experiment/CEMC/random activity maps for the best tau-MSE-CRPS trials.

The best trial for each method is the observed point closest to the normalized
ideal corner: high Kendall tau, low MSE, and low mean scaled CRPS. MC/CEMC CRPS
is taken at the same selected temperature as its tau and MSE. Random/RS CRPS is
temperature-independent.
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


def load_best_trials(score_dir: Path) -> pd.DataFrame:
    cemc = pd.read_csv(score_dir / "best_trial_cemc_crps.csv")
    random = pd.read_csv(score_dir / "best_trial_random_crps.csv")
    if cemc.empty or random.empty:
        raise SystemExit(f"Missing best-trial CSVs in {score_dir}")
    cemc = cemc.copy(); cemc["method"] = "cemc"
    random = random.copy(); random["method"] = "random"
    df = pd.concat([cemc, random], ignore_index=True, sort=False)
    for col in ["trial", "temperature", "tau", "mse", "crps", "selected_temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trial", "tau", "mse", "crps"])
    df["trial"] = df["trial"].astype(int)
    df["score_tau_minus_mse"] = df["tau"] - df["mse"]
    if "selected_temperature" not in df.columns and "temperature" in df.columns:
        df = df.rename(columns={"temperature": "selected_temperature"})
    if "selected_temperature" not in df.columns:
        df["selected_temperature"] = np.nan
    return df.sort_values(["method", "trial"]).reset_index(drop=True)


def filter_metrics_by_trial(metrics: pd.DataFrame, max_trial: int) -> pd.DataFrame:
    if max_trial <= 0:
        return metrics
    out = metrics[metrics["trial"] < max_trial].copy()
    if out.empty:
        raise SystemExit(f"No completed trials with trial < {max_trial} were found.")
    return out.reset_index(drop=True)


def points_for_best(metrics: pd.DataFrame, best_for: BestFor) -> pd.DataFrame:
    pts = metrics.copy()
    if best_for in {"cemc", "random"}:
        pts = pts[pts["method"] == best_for].copy()
    pts = pts.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau", "mse", "crps"])
    if pts.empty:
        raise SystemExit("No finite tau/MSE/CRPS points were found.")
    return pts.sort_values(["method", "trial"]).reset_index(drop=True)


def select_ideal_corner(points: pd.DataFrame) -> pd.Series:
    mse = points["mse"].astype(float)
    tau = points["tau"].astype(float)
    crps = points["crps"].astype(float)
    mse_range = max(float(mse.max() - mse.min()), 1.0e-15)
    tau_range = max(float(tau.max() - tau.min()), 1.0e-15)
    crps_range = max(float(crps.max() - crps.min()), 1.0e-15)
    work = points.copy()
    work["ideal_corner_distance_tau_mse_crps"] = np.sqrt(
        ((mse - float(mse.min())) / mse_range) ** 2 +
        ((float(tau.max()) - tau) / tau_range) ** 2 +
        ((crps - float(crps.min())) / crps_range) ** 2
    )
    return work.sort_values(
        ["ideal_corner_distance_tau_mse_crps", "mse", "crps", "tau"],
        ascending=[True, True, True, False],
    ).iloc[0]


def select_best_trials(points: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for method in ["cemc", "random"]:
        method_points = points[points["method"] == method]
        if method_points.empty:
            continue
        selected.append(select_ideal_corner(method_points))
    if not selected:
        raise SystemExit("No method-specific best trials could be selected.")
    return pd.DataFrame(selected).reset_index(drop=True)


def point_density_3d(points: pd.DataFrame, bins: int) -> np.ndarray:
    values = points[["tau", "mse", "crps"]].to_numpy(float)
    hist, edges = np.histogramdd(values, bins=bins)
    indices = []
    for axis in range(3):
        idx = np.searchsorted(edges[axis], values[:, axis], side="right") - 1
        indices.append(np.clip(idx, 0, bins - 1))
    return hist[tuple(indices)].astype(int)


def plot_selection_3d(
    points: pd.DataFrame,
    selected_by_method: pd.DataFrame,
    output: Path,
    density_bins: int,
    dpi: int,
) -> None:
    methods = [method for method in ("cemc", "random") if method in set(points["method"])]
    fig = plt.figure(figsize=(8.0 * len(methods), 7.0))
    for panel, method in enumerate(methods, start=1):
        method_points = points[points["method"] == method].copy()
        selected = selected_by_method[selected_by_method["method"] == method].iloc[0]
        density = point_density_3d(method_points, density_bins)
        ax = fig.add_subplot(1, len(methods), panel, projection="3d")
        scatter = ax.scatter(
            method_points["tau"], method_points["mse"], method_points["crps"],
            c=density, s=14.0 + 7.0 * np.sqrt(density), cmap="viridis",
            alpha=0.65, edgecolors="none",
        )
        ax.scatter(
            [selected["tau"]], [selected["mse"]], [selected["crps"]],
            marker="*", s=360, color="#d62728", edgecolor="white", linewidth=1.1,
            label=f"Selected trial {int(selected['trial'])}",
        )
        ax.scatter(
            [method_points["tau"].max()], [method_points["mse"].min()],
            [method_points["crps"].min()], marker="D", s=85,
            facecolor="none", edgecolor="black", linewidth=1.2,
            label="Normalized ideal corner",
        )
        ax.set_xlabel("Kendall tau")
        ax.set_ylabel("MSE")
        ax.set_zlabel("Mean scaled CRPS")
        if method == "cemc":
            temperature = int(round(float(selected["selected_temperature"])))
            title = (
                f"MC/CEMC trial points (n={len(method_points):,})\n"
                f"Selected trial {int(selected['trial'])} at {temperature} K"
            )
        else:
            title = (
                f"Random/RS trial points (n={len(method_points):,})\n"
                f"Selected trial {int(selected['trial'])}"
            )
        ax.set_title(title, weight="bold", pad=12)
        ax.legend(frameon=False, fontsize=9, loc="best")
        fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.68, label="3D-bin trial count")
    fig.suptitle(
        "Best-trial selection in normalized Tau/MSE/CRPS space",
        fontsize=16, weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def activity_columns(activity_scale: str) -> tuple[str, str, str]:
    if activity_scale == "scaled":
        return "experimental_activity_scaled", "cemc_pred_activity_scaled", "random_pred_activity_scaled"
    if activity_scale == "raw":
        return "experimental_activity", "cemc_pred_activity_mean", "random_pred_activity_mean"
    raise SystemExit(f"Unknown activity scale: {activity_scale}")


def load_activity_for_trials(
    results_root: Path, trials: set[int], world_size: int | None
) -> dict[int, pd.DataFrame]:
    df = read_sharded_csv(results_root, "predicted_activity_selected_by_trial.csv", world_size)
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    df = df[df["trial"].isin(trials)].copy()
    if df.empty:
        raise SystemExit(f"No predicted activity rows found for trials {sorted(trials)}")
    df["trial"] = df["trial"].astype(int)
    df["composition_index"] = pd.to_numeric(df["composition_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["composition_index"]).copy()
    df["composition_index"] = df["composition_index"].astype(int)
    df = df.drop_duplicates(subset=["trial", "composition_index"], keep="last")
    found = set(df["trial"].unique())
    missing = sorted(trials - found)
    if missing:
        raise SystemExit(f"No predicted activity rows found for trials {missing}")
    return {
        trial: part.sort_values("composition_index").reset_index(drop=True)
        for trial, part in df.groupby("trial")
    }


def vector_from_trial(df: pd.DataFrame, column: str, n_valid: int) -> np.ndarray:
    out = np.full(n_valid, np.nan, dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    comp = df["composition_index"].to_numpy(dtype=int)
    ok = (comp >= 0) & (comp < n_valid)
    out[comp[ok]] = values[ok]
    return out


def to_masked_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = ~mask.astype(bool)
    n_valid = int(np.count_nonzero(valid))
    if len(values) != n_valid:
        raise SystemExit(f"Activity vector length {len(values)} does not match unmasked grid cells {n_valid}")
    grid = np.full(mask.shape, np.nan, dtype=float)
    grid[valid] = values
    return grid


def plot_maps(
    grids: dict[str, np.ndarray],
    selected_by_method: pd.DataFrame,
    output: Path,
    value_label: str,
    cmap: str,
) -> None:
    selected_lookup = {str(row["method"]): row for _, row in selected_by_method.iterrows()}
    cemc = selected_lookup.get("cemc")
    random = selected_lookup.get("random")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=180, constrained_layout=True)
    titles = [
        "Experiment",
        "MC/CEMC" if cemc is None else f"MC/CEMC best trial {int(cemc['trial'])} at {int(round(float(cemc['selected_temperature'])))} K\nTau={float(cemc['tau']):.4g}, MSE={float(cemc['mse']):.4g}, CRPS={float(cemc['crps']):.4g}",
        "Random" if random is None else f"Random best trial {int(random['trial'])}\nTau={float(random['tau']):.4g}, MSE={float(random['mse']):.4g}, CRPS={float(random['crps']):.4g}",
    ]
    keys = ["experiment", "cemc", "random"]
    image = None
    for ax, key, title in zip(axes, keys, titles):
        image = ax.imshow(grids[key], cmap=cmap, vmin=-1.0, vmax=0.0)
        ax.set_title(title, weight="bold", pad=8)
        ax.axis("off")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(value_label, rotation=270, labelpad=15)
    cbar.set_ticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
    title_parts = []
    if cemc is not None:
        title_parts.append(f"MC best: trial {int(cemc['trial'])} at {int(round(float(cemc['selected_temperature'])))} K, tau={float(cemc['tau']):.4g}, MSE={float(cemc['mse']):.4g}, CRPS={float(cemc['crps']):.4g}")
    if random is not None:
        title_parts.append(f"Random best: trial {int(random['trial'])}, tau={float(random['tau']):.4g}, MSE={float(random['mse']):.4g}, CRPS={float(random['crps']):.4g}")
    fig.suptitle(" | ".join(title_parts), weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing shard_XX results")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--activity-scale", choices=["scaled", "raw"], default="scaled")
    parser.add_argument("--best-score-dir", default=None)
    parser.add_argument("--mask", default="/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy")
    parser.add_argument("--best-for", choices=["cemc", "random", "either"], default="either")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cmap", default="bwr_r")
    parser.add_argument("--density-bins", type=int, default=15, help="Bins per axis for 3D scatter density coloring")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-trial", type=int, default=1000, help="Only use trials with trial < this value for best-trial selection. Use 0 for all completed trials.")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("results/best_activity_maps_tau_mse_crps" if args.activity_scale == "scaled" else "results/best_activity_maps_tau_mse_crps_raw").resolve()
    mask = np.load(args.mask).astype(bool)
    n_valid = int(np.count_nonzero(~mask))
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else Path("results/crps_selection" if args.activity_scale == "scaled" else "results/crps_selection_raw").resolve()

    metrics = filter_metrics_by_trial(
        load_best_trials(score_dir),
        args.max_trial,
    )
    points = points_for_best(metrics, args.best_for)
    selected_by_method = select_best_trials(points)
    selected_lookup = {str(row["method"]): row for _, row in selected_by_method.iterrows()}
    cemc_selected = selected_lookup.get("cemc")
    random_selected = selected_lookup.get("random")
    if cemc_selected is None and random_selected is None:
        raise SystemExit("No MC/CEMC or Random best trial was selected.")
    experiment_trial = int((cemc_selected if cemc_selected is not None else random_selected)["trial"])
    cemc_trial = int((cemc_selected if cemc_selected is not None else random_selected)["trial"])
    random_trial = int((random_selected if random_selected is not None else cemc_selected)["trial"])

    activity_by_trial = load_activity_for_trials(
        results_root, {experiment_trial, cemc_trial, random_trial}, args.world_size
    )
    experiment_acts = activity_by_trial[experiment_trial]
    cemc_acts = activity_by_trial[cemc_trial]
    random_acts = activity_by_trial[random_trial]
    exp_col, cemc_col, random_col = activity_columns(args.activity_scale)
    grids = {
        "experiment": to_masked_grid(vector_from_trial(experiment_acts, exp_col, n_valid), mask),
        "cemc": to_masked_grid(vector_from_trial(cemc_acts, cemc_col, n_valid), mask),
        "random": to_masked_grid(vector_from_trial(random_acts, random_col, n_valid), mask),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    map_png = output_dir / f"best_tau_mse_crps_cemc_trial_{cemc_trial:04d}_random_trial_{random_trial:04d}_activity_maps.png"
    cemc_values_csv = output_dir / f"trial_{cemc_trial:04d}_cemc_best_activity_map_values.csv"
    random_values_csv = output_dir / f"trial_{random_trial:04d}_random_best_activity_map_values.csv"
    selected_csv = output_dir / "selected_tau_mse_crps_trials_by_method.csv"
    selection_3d_png = output_dir / "tau_mse_crps_best_trial_selection_3d.png"

    plot_maps(grids, selected_by_method, map_png, "Scaled activity (-1 high, 0 low)" if args.activity_scale == "scaled" else "Raw activity", args.cmap)
    plot_selection_3d(
        points, selected_by_method, selection_3d_png, args.density_bins, args.dpi
    )
    cemc_acts.to_csv(cemc_values_csv, index=False)
    random_acts.to_csv(random_values_csv, index=False)
    selected_by_method.to_csv(selected_csv, index=False)
    for key, grid in grids.items():
        trial_for_grid = experiment_trial if key == "experiment" else cemc_trial if key == "cemc" else random_trial
        np.save(output_dir / f"trial_{trial_for_grid:04d}_{key}_grid.npy", grid)

    for _, selected in selected_by_method.iterrows():
        print(
            f"selected method={selected['method']} trial={int(selected['trial'])} "
            f"tau={float(selected['tau']):.6g} mse={float(selected['mse']):.6g} "
            f"crps={float(selected['crps']):.6g}"
        )
    print(f"wrote {map_png}")
    print(f"wrote {selection_3d_png}")
    print(f"wrote {selected_csv}")


if __name__ == "__main__":
    main()
