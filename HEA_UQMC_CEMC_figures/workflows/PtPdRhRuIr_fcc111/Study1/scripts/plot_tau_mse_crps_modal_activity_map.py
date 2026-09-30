#!/usr/bin/env python3
"""Select and plot an activity map from the modal 3D tau-MSE-CRPS bin.

MC/CEMC uses every available (trial, temperature) point. Random/RS uses one
temperature-independent point per trial. Separate regular 3D histograms and
representatives are computed before their activity maps are combined.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_sharded_csv(results_root: Path, filename: str, world_size: int | None) -> pd.DataFrame:
    shards = (sorted(results_root.glob("shard_*")) if world_size is None else
              [results_root / f"shard_{i:02d}" for i in range(world_size)])
    frames = []
    for shard in shards:
        path = shard / filename
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if not df.empty:
                df["source_shard"] = shard.name
                frames.append(df)
    if not frames:
        raise SystemExit(f"No {filename} files found under {results_root}")
    return pd.concat(frames, ignore_index=True)


def filter_trials(df: pd.DataFrame, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    out = df.copy()
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out = out.dropna(subset=["trial"])
    out["trial"] = out["trial"].astype(int)
    if trial_min is not None:
        out = out[out["trial"] >= trial_min]
    if trial_max is not None:
        out = out[out["trial"] <= trial_max]
    return out


def load_score_points(score_dir: Path, trial_min: int | None, trial_max: int | None) -> dict[str, pd.DataFrame]:
    cemc = pd.read_csv(score_dir / "crps_by_trial_temperature.csv")
    random = pd.read_csv(score_dir / "random_crps_by_trial.csv")
    cemc = filter_trials(cemc, trial_min, trial_max)
    random = filter_trials(random, trial_min, trial_max)
    for col in ("trial", "temperature", "tau", "mse", "crps"):
        if col in cemc.columns:
            cemc[col] = pd.to_numeric(cemc[col], errors="coerce")
    for col in ("trial", "tau", "mse", "crps"):
        if col in random.columns:
            random[col] = pd.to_numeric(random[col], errors="coerce")
    cemc = cemc.dropna(subset=["trial", "temperature", "tau", "mse", "crps"]).copy()
    random = random.dropna(subset=["trial", "tau", "mse", "crps"]).copy()
    cemc["trial"] = cemc["trial"].astype(int)
    cemc["temperature"] = cemc["temperature"].astype(int)
    cemc["method"] = "cemc"
    random["trial"] = random["trial"].astype(int)
    random["temperature"] = np.nan
    random["method"] = "random"
    return {
        "cemc": cemc.sort_values(["trial", "temperature"]).reset_index(drop=True),
        "random": random.sort_values("trial").reset_index(drop=True),
    }


def modal_histogram3d(points: pd.DataFrame, bins: int) -> tuple[pd.Series, pd.DataFrame, np.ndarray, list[np.ndarray]]:
    columns = ["tau", "mse", "crps"]
    values = points[columns].to_numpy(float)
    hist, edges = np.histogramdd(values, bins=bins)
    indices = []
    for axis in range(3):
        idx = np.searchsorted(edges[axis], values[:, axis], side="right") - 1
        indices.append(np.clip(idx, 0, bins - 1))
    ix, iy, iz = indices
    counts = hist[ix, iy, iz].astype(int)
    centers_by_axis = [(edge[:-1] + edge[1:]) / 2.0 for edge in edges]
    centers = np.column_stack([
        centers_by_axis[0][ix], centers_by_axis[1][iy], centers_by_axis[2][iz]
    ])

    assigned = points.copy()
    assigned[["tau_bin", "mse_bin", "crps_bin"]] = np.column_stack(indices)
    assigned[["bin_center_tau", "bin_center_mse", "bin_center_crps"]] = centers
    assigned["bin_count"] = counts
    max_count = int(hist.max())
    modal_candidates = np.argwhere(hist == max_count)

    tie_rows = []
    for candidate in modal_candidates:
        mask = (ix == candidate[0]) & (iy == candidate[1]) & (iz == candidate[2])
        members = assigned.loc[mask]
        tie_rows.append({
            "tau_bin": int(candidate[0]), "mse_bin": int(candidate[1]), "crps_bin": int(candidate[2]),
            "mean_tau": float(members["tau"].mean()),
            "mean_mse": float(members["mse"].mean()),
            "mean_crps": float(members["crps"].mean()),
        })
    chosen = pd.DataFrame(tie_rows).sort_values(
        ["mean_tau", "mean_mse", "mean_crps", "tau_bin", "mse_bin", "crps_bin"],
        ascending=[False, True, True, True, True, True],
    ).iloc[0]
    modal_mask = (
        (assigned["tau_bin"] == int(chosen["tau_bin"])) &
        (assigned["mse_bin"] == int(chosen["mse_bin"])) &
        (assigned["crps_bin"] == int(chosen["crps_bin"]))
    )
    assigned["is_modal_bin"] = modal_mask
    ranges = np.maximum(np.ptp(values, axis=0), 1.0e-15)
    assigned["normalized_distance_to_bin_center"] = np.sqrt(
        np.sum(((values - centers) / ranges) ** 2, axis=1)
    )
    representative = assigned[modal_mask].sort_values(
        ["normalized_distance_to_bin_center", "tau", "mse", "crps", "trial"],
        ascending=[True, False, True, True, True],
    ).iloc[0].copy()
    representative["modal_bin_count"] = max_count
    representative["n_tied_modal_bins"] = len(modal_candidates)
    representative["histogram_bins_per_axis"] = bins
    return representative, assigned, hist, edges


def plot_histograms3d(
    histograms: dict[str, np.ndarray], edge_sets: dict[str, list[np.ndarray]],
    representatives: pd.DataFrame,
    output: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(16.0, 7.2))
    for panel, method in enumerate(["cemc", "random"], start=1):
        hist, edges = histograms[method], edge_sets[method]
        rep = representatives[representatives["method"] == method].iloc[0]
        occupied = np.argwhere(hist > 0)
        centers = [(edge[:-1] + edge[1:]) / 2.0 for edge in edges]
        tau = centers[0][occupied[:, 0]]
        mse = centers[1][occupied[:, 1]]
        crps = centers[2][occupied[:, 2]]
        count = hist[tuple(occupied.T)]
        sizes = 15.0 + 150.0 * count / count.max()
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        sc = ax.scatter(tau, mse, crps, c=count, s=sizes, cmap="viridis", alpha=0.72)
        ax.scatter(rep["bin_center_tau"], rep["bin_center_mse"], rep["bin_center_crps"],
                   marker="*", s=280, color="#d62728", edgecolor="white", linewidth=0.8,
                   label=f"Modal bin (n={int(rep['modal_bin_count'])})")
        temp = "" if method == "random" else f", {int(rep['temperature'])} K"
        ax.scatter(rep["tau"], rep["mse"], rep["crps"], marker="X", s=140, color="black",
                   label=f"Trial {int(rep['trial'])}{temp}")
        ax.set_xlabel("Kendall tau"); ax.set_ylabel("MSE"); ax.set_zlabel("Mean scaled CRPS")
        n_points = int((hist * 1).sum())
        ax.set_title(f"{'MC/CEMC trial-temperature' if method == 'cemc' else 'Random/RS trial'} points (n={n_points:,})")
        ax.legend(frameon=False, fontsize=9)
        fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.70, label="Point count")
    fig.suptitle("Independent modal 3D tau/MSE/CRPS histograms", weight="bold", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def activity_columns(activity_scale: str) -> tuple[str, str, str]:
    if activity_scale == "scaled":
        return "experimental_activity_scaled", "cemc_pred_activity_scaled", "random_pred_activity_scaled"
    if activity_scale == "raw":
        return "experimental_activity", "cemc_pred_activity_mean", "random_pred_activity_mean"
    raise SystemExit(f"Unknown activity scale: {activity_scale}")


def load_activity_for_trial(results_root: Path, trial: int, world_size: int | None) -> pd.DataFrame:
    df = filter_trials(
        read_sharded_csv(results_root, "predicted_activity_selected_by_trial.csv", world_size),
        trial, trial,
    )
    if df.empty:
        raise SystemExit(f"No predicted activity rows found for trial {trial}")
    df["composition_index"] = pd.to_numeric(df["composition_index"], errors="coerce")
    df = df.dropna(subset=["composition_index"])
    df["composition_index"] = df["composition_index"].astype(int)
    return df.drop_duplicates("composition_index", keep="last").sort_values("composition_index")


def load_mc_activity_from_zarr(
    results_root: Path, source_shard: str, trial: int, temperature: int, activity_scale: str
) -> np.ndarray:
    store = results_root / source_shard / "trial_zarr" / f"trial_{trial:06d}" / "mc.zarr"
    with open(store / ".zattrs", encoding="utf-8") as handle:
        attrs = json.load(handle)
    temperatures = [int(x) for x in attrs["temperatures"]]
    if temperature not in temperatures:
        raise SystemExit(f"Temperature {temperature} K is absent from {store}")
    with open(store / "activity_mean" / ".zarray", encoding="utf-8") as handle:
        meta = json.load(handle)
    shape = tuple(int(x) for x in meta["shape"])
    dtype = np.dtype(str(meta["dtype"]))
    raw = np.fromfile(store / "activity_mean" / "0.0", dtype=dtype)
    if raw.size != int(np.prod(shape)):
        raise SystemExit(f"Unexpected activity_mean chunk size in {store}")
    values = raw.reshape(shape)[temperatures.index(temperature)].astype(float)
    if activity_scale == "raw":
        return values
    den = float(np.nanmax(values) - np.nanmin(values))
    return np.zeros_like(values) if abs(den) < 1e-30 else -(values - np.nanmin(values)) / den


def vector_from_trial(df: pd.DataFrame, column: str, n_valid: int) -> np.ndarray:
    comp = df["composition_index"].to_numpy(int)
    actual = np.sort(np.unique(comp))
    if len(actual) != n_valid or not np.array_equal(actual, np.arange(n_valid)):
        raise SystemExit(f"Mask/data mismatch: expected composition indices 0..{n_valid - 1}")
    out = np.full(n_valid, np.nan)
    out[comp] = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
    return out


def to_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, np.nan)
    grid[~mask] = values
    return grid


def plot_maps(grids: dict[str, np.ndarray], reps: pd.DataFrame, output: Path, cmap: str, dpi: int) -> None:
    mc = reps[reps["method"] == "cemc"].iloc[0]
    rs = reps[reps["method"] == "random"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    titles = [
        "Experiment",
        f"MC/CEMC: trial {int(mc['trial'])}\n"
        f"T={int(mc['temperature'])} K, bin n={int(mc['modal_bin_count'])}\n"
        f"tau={mc['tau']:.4f}, MSE={mc['mse']:.4f}, CRPS={mc['crps']:.4f}",
        f"Random/RS: trial {int(rs['trial'])}\n"
        f"bin n={int(rs['modal_bin_count'])}\n"
        f"tau={rs['tau']:.4f}, MSE={rs['mse']:.4f}, CRPS={rs['crps']:.4f}",
    ]
    image = None
    for ax, key, title in zip(axes, ["experiment", "cemc", "random"], titles):
        image = ax.imshow(grids[key], cmap=cmap, vmin=-1.0, vmax=0.0)
        ax.set_title(title, weight="bold", fontsize=10.5, pad=8)
        ax.axis("off")
    cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cb.set_label("Scaled activity (-1 high, 0 low)", rotation=270, labelpad=15)
    fig.suptitle(
        "Independent representatives from modal 3D tau/MSE/CRPS bins", weight="bold"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--activity-scale", choices=["scaled", "raw"], default="scaled")
    parser.add_argument("--best-score-dir", default=None)
    parser.add_argument("--bins", type=int, default=20, help="Regular histogram bins per axis")
    parser.add_argument("--mask", default="/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cmap", default="bwr_r")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("results/modal_tau_mse_crps_activity_map" if args.activity_scale == "scaled" else "results/modal_tau_mse_crps_activity_map_raw").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else Path("results/crps_selection" if args.activity_scale == "scaled" else "results/crps_selection_raw").resolve()
    points_by_method = load_score_points(score_dir, args.trial_min, args.trial_max)
    rep_rows, assignment_rows, histograms, edge_sets = [], [], {}, {}
    for method, points in points_by_method.items():
        rep, assigned, hist, edges = modal_histogram3d(points, args.bins)
        rep_rows.append(rep); assignment_rows.append(assigned)
        histograms[method] = hist; edge_sets[method] = edges
    representatives = pd.DataFrame(rep_rows).reset_index(drop=True)
    assignments = pd.concat(assignment_rows, ignore_index=True)
    mc = representatives[representatives["method"] == "cemc"].iloc[0]
    rs = representatives[representatives["method"] == "random"].iloc[0]
    mc_trial, rs_trial = int(mc["trial"]), int(rs["trial"])
    mc_selected_activities = load_activity_for_trial(results_root, mc_trial, args.world_size)
    rs_activities = load_activity_for_trial(results_root, rs_trial, args.world_size)
    mask = np.load(args.mask).astype(bool)
    n_valid = int(np.count_nonzero(~mask))
    exp_col, cemc_col, random_col = activity_columns(args.activity_scale)
    mc_values = load_mc_activity_from_zarr(
        results_root, str(mc["source_shard"]), mc_trial, int(mc["temperature"]), args.activity_scale
    )
    grids = {
        "experiment": to_grid(vector_from_trial(mc_selected_activities, exp_col, n_valid), mask),
        "cemc": to_grid(mc_values, mask),
        "random": to_grid(vector_from_trial(rs_activities, random_col, n_valid), mask),
    }

    hist_png = output_dir / "tau_mse_crps_3d_histogram.png"
    map_png = output_dir / f"mc_trial_{mc_trial:04d}_random_trial_{rs_trial:04d}_modal_tau_mse_crps_activity_maps.png"
    rep_csv = output_dir / "modal_tau_mse_crps_representative_trials.csv"
    assignments_csv = output_dir / "tau_mse_crps_trial_bin_assignments.csv"
    mc_values_csv = output_dir / f"trial_{mc_trial:04d}_mc_activity_map_values.csv"
    rs_values_csv = output_dir / f"trial_{rs_trial:04d}_random_activity_map_values.csv"
    plot_histograms3d(histograms, edge_sets, representatives, hist_png, args.dpi)
    plot_maps(grids, representatives, map_png, args.cmap, args.dpi)
    representatives.to_csv(rep_csv, index=False)
    assignments.to_csv(assignments_csv, index=False)
    cemc_value_col = "cemc_pred_activity_scaled" if args.activity_scale == "scaled" else "cemc_pred_activity_mean"
    random_value_col = "random_pred_activity_scaled" if args.activity_scale == "scaled" else "random_pred_activity_mean"
    pd.DataFrame({"composition_index": np.arange(len(mc_values)), cemc_value_col: mc_values}).to_csv(mc_values_csv, index=False)
    rs_activities.to_csv(rs_values_csv, index=False)
    for key, grid in grids.items():
        grid_trial = mc_trial if key != "random" else rs_trial
        np.save(output_dir / f"trial_{grid_trial:04d}_{key}_grid.npy", grid)

    print(
        f"selected MC trial={mc_trial} T={int(mc['temperature'])} K modal_count={int(mc['modal_bin_count'])}; "
        f"Random trial={rs_trial} modal_count={int(rs['modal_bin_count'])}"
    )
    for path in [hist_png, map_png, rep_csv, assignments_csv, mc_values_csv, rs_values_csv]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
