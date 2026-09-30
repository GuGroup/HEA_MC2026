#!/usr/bin/env python3
"""Select representative MC/CEMC and random trials from modal tau-MSE hexbins.

For each method, the script draws a trial-wise MSE-x/tau-y hexbin, finds the
hexagon with the largest count, and selects the trial closest to that hexagon's
center in normalized MSE/tau coordinates. MC/CEMC and random are selected
independently. Activity maps and machine-readable selection diagnostics are
saved for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


def read_sharded_csv(results_root: Path, filename: str, world_size: int | None) -> pd.DataFrame:
    shard_dirs = (
        sorted(results_root.glob("shard_*"))
        if world_size is None
        else [results_root / f"shard_{idx:02d}" for idx in range(world_size)]
    )
    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_shard"] = shard_dir.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"No {filename} files found under {results_root}")
    return pd.concat(frames, ignore_index=True)


def filter_trials(df: pd.DataFrame, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    out = df.copy()
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out = out.dropna(subset=["trial"]).copy()
    out["trial"] = out["trial"].astype(int)
    if trial_min is not None:
        out = out[out["trial"] >= trial_min]
    if trial_max is not None:
        out = out[out["trial"] <= trial_max]
    return out


def load_score_points(score_dir: Path, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    cemc = pd.read_csv(score_dir / "crps_by_trial_temperature.csv")
    random = pd.read_csv(score_dir / "random_crps_by_trial.csv")
    if "trial" not in cemc.columns or "trial" not in random.columns:
        raise SystemExit(f"Missing score CSVs under {score_dir}")
    cemc = filter_trials(cemc, trial_min, trial_max)
    random = filter_trials(random, trial_min, trial_max)
    required_cemc = {"trial", "temperature", "tau", "mse", "crps"}
    required_random = {"trial", "tau", "mse", "crps"}
    missing = sorted(required_cemc - set(cemc.columns))
    if missing:
        raise SystemExit(f"CEMC score CSV is missing: {', '.join(missing)}")
    missing = sorted(required_random - set(random.columns))
    if missing:
        raise SystemExit(f"Random score CSV is missing: {', '.join(missing)}")
    cemc = cemc.copy()
    cemc["method"] = "cemc"
    random = random.copy()
    random["method"] = "random"
    if "selected_temperature" not in cemc.columns:
        cemc = cemc.rename(columns={"temperature": "selected_temperature"})
    for df in (cemc, random):
        for col in ["trial", "tau", "mse", "crps"] + (["selected_temperature"] if "selected_temperature" in df.columns else []):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.concat([cemc, random], ignore_index=True, sort=False)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["trial", "tau", "mse", "crps"])
    out["trial"] = out["trial"].astype(int)
    out = out.sort_values(["method", "trial", "selected_temperature"] if "selected_temperature" in out.columns else ["method", "trial"])
    return out.reset_index(drop=True)


def method_points(metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    out = metrics[metrics["method"].eq(method)].copy()
    if out.empty:
        return out
    out["score_tau_minus_mse"] = out["tau"] - out["mse"]
    if "selected_temperature" not in out.columns:
        out["selected_temperature"] = np.nan
    return out[["trial", "method", "tau", "mse", "crps", "score_tau_minus_mse", "selected_temperature", "source_shard"] if "source_shard" in out.columns else ["trial", "method", "tau", "mse", "crps", "score_tau_minus_mse", "selected_temperature"]]


def modal_hexbin(points: pd.DataFrame, gridsize: int) -> tuple[pd.Series, pd.DataFrame]:
    """Return modal-bin representative and nearest-center trial assignments.

    Matplotlib supplies the displayed hex centers and counts. Trials are assigned
    to their nearest occupied center after scaling x and y by the same grid
    dimensions used for hexbin. This makes the representative an observed trial,
    not a synthetic MSE/tau coordinate.
    """
    fig, ax = plt.subplots()
    hb = ax.hexbin(
        points["mse"], points["tau"], gridsize=gridsize, mincnt=1,
        cmap="viridis", linewidths=0.2,
    )
    centers = np.asarray(hb.get_offsets(), dtype=float)
    counts = np.asarray(hb.get_array(), dtype=float)
    plt.close(fig)
    if len(centers) == 0:
        raise SystemExit(f"No occupied hexbin was found for {points['method'].iloc[0]}")

    # Reproduce Matplotlib's Axes.hexbin bin assignment exactly. A simple
    # nearest-center calculation misclassifies some points on hex boundaries.
    x = points["mse"].to_numpy(float)
    y = points["tau"].to_numpy(float)
    nx = int(gridsize)
    ny = int(nx / np.sqrt(3.0))
    nx1, ny1 = nx + 1, ny + 1
    nx2, ny2 = nx, ny
    xmin, xmax = mtransforms.nonsingular(float(x.min()), float(x.max()), expander=0.1)
    ymin, ymax = mtransforms.nonsingular(float(y.min()), float(y.max()), expander=0.1)
    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny
    ix = (x - xmin) / sx
    iy = (y - ymin) / sy
    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)
    i1 = ix1 * ny1 + iy1
    i2 = nx1 * ny1 + ix2 * ny2 + iy2
    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    use_first_lattice = d1 < d2
    full_index = np.where(use_first_lattice, i1, i2)
    distance = np.sqrt(np.where(use_first_lattice, d1, d2))

    n_full = nx1 * ny1 + nx2 * ny2
    full_counts = np.bincount(full_index, minlength=n_full)
    occupied_full_indices = np.flatnonzero(full_counts > 0)
    if len(occupied_full_indices) != len(centers):
        raise RuntimeError("Internal hexbin centers do not match reconstructed occupied bins")
    full_to_visible = np.full(n_full, -1, dtype=int)
    full_to_visible[occupied_full_indices] = np.arange(len(occupied_full_indices))
    assignment = full_to_visible[full_index]
    reconstructed_counts = np.bincount(assignment, minlength=len(centers))
    if not np.array_equal(reconstructed_counts, counts.astype(int)):
        raise RuntimeError("Reconstructed trial-to-hexbin assignments do not match Matplotlib counts")

    assigned = points.copy().reset_index(drop=True)
    assigned["hex_index"] = assignment
    assigned["hex_center_mse"] = centers[assignment, 0]
    assigned["hex_center_tau"] = centers[assignment, 1]
    assigned["hex_count"] = counts[assignment].astype(int)
    assigned["normalized_distance_to_hex_center"] = distance

    max_count = int(np.max(counts))
    tied = np.flatnonzero(counts == max_count)
    tie_rows = []
    for idx in tied:
        members = assigned[assigned["hex_index"] == idx]
        tie_rows.append(
            {
                "hex_index": int(idx),
                "mean_score": float(members["score_tau_minus_mse"].mean()),
                "mean_tau": float(members["tau"].mean()),
                "mean_mse": float(members["mse"].mean()),
            }
        )
    chosen_hex = int(
        pd.DataFrame(tie_rows)
        .sort_values(["mean_score", "mean_tau", "mean_mse", "hex_index"], ascending=[False, False, True, True])
        .iloc[0]["hex_index"]
    )
    assigned["is_modal_hexbin"] = assigned["hex_index"] == chosen_hex
    members = assigned[assigned["is_modal_hexbin"]].copy()
    representative = members.sort_values(
        ["normalized_distance_to_hex_center", "score_tau_minus_mse", "tau", "mse", "trial"],
        ascending=[True, False, False, True, True],
    ).iloc[0].copy()
    representative["modal_hex_count"] = max_count
    representative["assigned_member_count"] = int(len(members))
    representative["n_tied_max_count_hexbins"] = int(len(tied))
    representative["hexbin_gridsize"] = int(gridsize)
    return representative, assigned


def plot_hexbins(
    points_by_method: dict[str, pd.DataFrame],
    representatives: pd.DataFrame,
    output: Path,
    gridsize: int,
    dpi: int,
) -> None:
    labels = {"cemc": "MC/CEMC selected", "random": "Random/RS"}
    cmaps = {"cemc": "Blues", "random": "Oranges"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), sharex=False, sharey=False)
    for ax, method in zip(axes, ["cemc", "random"]):
        pts = points_by_method[method]
        rep = representatives[representatives["method"] == method].iloc[0]
        hb = ax.hexbin(
            pts["mse"], pts["tau"], gridsize=gridsize, mincnt=1,
            cmap=cmaps[method], linewidths=0.2,
        )
        ax.scatter(
            [rep["hex_center_mse"]], [rep["hex_center_tau"]], marker="o", s=105,
            facecolor="none", edgecolor="black", linewidth=1.5,
            label=f"Modal hexbin, count={int(rep['modal_hex_count'])}",
        )
        ax.scatter(
            [rep["mse"]], [rep["tau"]], marker="*", s=190,
            color="#d62728", edgecolor="white", linewidth=0.7,
            label=f"Representative trial {int(rep['trial'])}", zorder=4,
        )
        ax.set_title(f"{labels[method]} tau/MSE hexbin")
        ax.set_xlabel("MSE")
        ax.set_ylabel("Kendall tau")
        ax.legend(frameon=False, fontsize=9)
        fig.colorbar(hb, ax=ax, label="Trial count")
    fig.suptitle("Modal tau-MSE hexbins and representative trials", weight="bold")
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
        trial,
        trial,
    )
    if df.empty:
        raise SystemExit(f"No predicted activity rows found for trial {trial}")
    df["composition_index"] = pd.to_numeric(df["composition_index"], errors="coerce")
    df = df.dropna(subset=["composition_index"]).copy()
    df["composition_index"] = df["composition_index"].astype(int)
    return df.drop_duplicates(subset=["composition_index"], keep="last").sort_values("composition_index")


def vector_from_trial(df: pd.DataFrame, column: str, n_valid: int) -> np.ndarray:
    if column not in df.columns:
        raise SystemExit(f"Missing activity column: {column}")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
    comp = df["composition_index"].to_numpy(int)
    expected = np.arange(n_valid, dtype=int)
    actual = np.sort(np.unique(comp))
    if len(actual) != n_valid or not np.array_equal(actual, expected):
        raise SystemExit(
            "Mask/data mismatch: the mask has "
            f"{n_valid} unmasked cells, but trial activity uses "
            f"{len(actual)} composition indices (expected 0..{n_valid - 1}). "
            "Use the 1400-composition static_grid_mask.npy."
        )
    out = np.full(n_valid, np.nan, dtype=float)
    out[comp] = values
    return out


def to_masked_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = ~mask.astype(bool)
    if len(values) != int(np.count_nonzero(valid)):
        raise SystemExit("Activity vector length does not match unmasked grid cells")
    grid = np.full(mask.shape, np.nan, dtype=float)
    grid[valid] = values
    return grid


def plot_maps(
    grids: dict[str, np.ndarray], representatives: pd.DataFrame, output: Path, cmap: str, dpi: int
) -> None:
    mc = representatives[representatives["method"] == "cemc"].iloc[0]
    rs = representatives[representatives["method"] == "random"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    titles = [
        "Experiment",
        f"MC modal trial {int(mc['trial'])}\nTau={mc['tau']:.4g}, MSE={mc['mse']:.4g}, bin n={int(mc['modal_hex_count'])}",
        f"RS modal trial {int(rs['trial'])}\nTau={rs['tau']:.4g}, MSE={rs['mse']:.4g}, bin n={int(rs['modal_hex_count'])}",
    ]
    image = None
    for ax, key, title in zip(axes, ["experiment", "cemc", "random"], titles):
        image = ax.imshow(grids[key], cmap=cmap, vmin=-1.0, vmax=0.0)
        ax.set_title(title, weight="bold", pad=8)
        ax.axis("off")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Scaled activity (-1 high, 0 low)", rotation=270, labelpad=15)
    cbar.set_ticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
    fig.suptitle("Representative activity maps from highest-count tau/MSE hexbins", weight="bold")
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
    parser.add_argument("--hexbin-gridsize", type=int, default=20)
    parser.add_argument("--mask", default="/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cmap", default="bwr_r")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.hexbin_gridsize < 2:
        raise SystemExit("--hexbin-gridsize must be at least 2")

    results_root = Path(args.results_root).resolve()
    default_output_dir = Path("results/modal_hexbin_activity_maps" if args.activity_scale == "scaled" else "results/modal_hexbin_activity_maps_raw")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    default_score_dir = Path("results/crps_selection" if args.activity_scale == "scaled" else "results/crps_selection_raw")
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else default_score_dir.resolve()
    mask = np.load(args.mask).astype(bool)
    n_valid = int(np.count_nonzero(~mask))

    metrics = load_score_points(score_dir, args.trial_min, args.trial_max)
    points = {method: method_points(metrics, method) for method in ["cemc", "random"]}
    selections = []
    assignments = []
    for method in ["cemc", "random"]:
        selected, assigned = modal_hexbin(points[method], args.hexbin_gridsize)
        selections.append(selected)
        assignments.append(assigned)
    representatives = pd.DataFrame(selections).reset_index(drop=True)
    trial_assignments = pd.concat(assignments, ignore_index=True)

    mc_trial = int(representatives.loc[representatives["method"] == "cemc", "trial"].iloc[0])
    rs_trial = int(representatives.loc[representatives["method"] == "random", "trial"].iloc[0])
    mc_acts = load_activity_for_trial(results_root, mc_trial, args.world_size)
    rs_acts = load_activity_for_trial(results_root, rs_trial, args.world_size)
    exp_col, cemc_col, random_col = activity_columns(args.activity_scale)
    grids = {
        "experiment": to_masked_grid(vector_from_trial(mc_acts, exp_col, n_valid), mask),
        "cemc": to_masked_grid(vector_from_trial(mc_acts, cemc_col, n_valid), mask),
        "random": to_masked_grid(vector_from_trial(rs_acts, random_col, n_valid), mask),
    }

    hex_png = output_dir / "tau_mse_modal_hexbins.png"
    map_png = output_dir / f"mc_trial_{mc_trial:04d}_rs_trial_{rs_trial:04d}_modal_hexbin_activity_maps.png"
    summary_csv = output_dir / "modal_hexbin_representative_trials.csv"
    assignments_csv = output_dir / "modal_hexbin_trial_assignments.csv"
    mc_values_csv = output_dir / f"trial_{mc_trial:04d}_mc_modal_activity_map_values.csv"
    rs_values_csv = output_dir / f"trial_{rs_trial:04d}_rs_modal_activity_map_values.csv"

    plot_hexbins(points, representatives, hex_png, args.hexbin_gridsize, args.dpi)
    plot_maps(grids, representatives, map_png, args.cmap, args.dpi)
    representatives.to_csv(summary_csv, index=False)
    trial_assignments.to_csv(assignments_csv, index=False)
    mc_acts.to_csv(mc_values_csv, index=False)
    rs_acts.to_csv(rs_values_csv, index=False)
    np.save(output_dir / f"trial_{mc_trial:04d}_experiment_grid.npy", grids["experiment"])
    np.save(output_dir / f"trial_{mc_trial:04d}_mc_grid.npy", grids["cemc"])
    np.save(output_dir / f"trial_{rs_trial:04d}_rs_grid.npy", grids["random"])

    for _, row in representatives.iterrows():
        print(
            f"selected method={row['method']} trial={int(row['trial'])} "
            f"modal_count={int(row['modal_hex_count'])} "
            f"tau={row['tau']:.6g} mse={row['mse']:.6g}"
        )
    for path in [hex_png, map_png, summary_csv, assignments_csv, mc_values_csv, rs_values_csv]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
