#!/usr/bin/env python3
"""KDE-smoothed 3D tau/MSE/CRPS selection and activity-map plotting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

COMPACT = False

def read_sharded_csv(results_root: Path, filename: str, world_size: int | None) -> pd.DataFrame:
    shards = (
        sorted(results_root.glob("shard_*"))
        if world_size is None
        else [results_root / f"shard_{i:02d}" for i in range(world_size)]
    )
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
    out = out.dropna(subset=["trial"]).copy()
    out["trial"] = out["trial"].astype(int)
    if trial_min is not None:
        out = out[out["trial"] >= trial_min]
    if trial_max is not None:
        out = out[out["trial"] <= trial_max]
    return out


def signed_log1p(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    x = values[finite]
    out[finite] = np.sign(x) * np.log1p(np.abs(x))
    return out

def clip_outliers(values: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Clips values to the specified percentiles to remove extreme outliers."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values
    
    finite_values = values[finite]
    lower_bound = np.percentile(finite_values, lower_percentile)
    upper_bound = np.percentile(finite_values, upper_percentile)
    
    clipped_values = values.copy()
    clipped_values[finite] = np.clip(finite_values, lower_bound, upper_bound)
    return clipped_values

def scale_prediction_to_minus1_0(values: np.ndarray) -> np.ndarray:
    """Match the existing prediction scaling: minimum -> 0, maximum -> -1."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if not finite.any():
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    den = hi - lo
    out[finite] = 0.0 if abs(den) < 1.0e-30 else -(values[finite] - lo) / den
    return out


def load_matched_activity(path: Path, n_valid: int) -> np.ndarray:
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    expected = [str(i) for i in range(n_valid)]
    missing = [key for key in expected if key not in records]
    if missing:
        raise SystemExit(f"{path} is missing composition indices; first missing index: {missing[0]}")
    values = np.asarray([records[key] for key in expected], dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"{path} contains non-finite activity values")
    return values


def load_trial_shifts(be_path: Path, composition_path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """Load trial-level BE and composition shifts keyed by element."""
    be = pd.read_csv(be_path).set_index("trial")
    composition = pd.read_csv(composition_path).set_index("trial")
    elements = ("Ir", "Pd", "Pt", "Rh", "Ru")
    trials = set(be.index.astype(int)) & set(composition.index.astype(int))
    result: dict[int, dict[str, dict[str, float]]] = {}
    for trial in trials:
        result[int(trial)] = {
            "be": {element: float(be.loc[trial, f"be_shift_{element}"]) for element in elements},
            "composition": {
                element: float(composition.loc[trial, f"comp_shift_{element}"])
                for element in elements
            },
        }
    return result


def format_trial_shifts(label: str, trial: int, shifts: dict[int, dict[str, dict[str, float]]]) -> str:
    if trial not in shifts:
        raise SystemExit(f"Trial {trial} is absent from the BE/composition shift tables")
    record = shifts[trial]
    be_text = ", ".join(f"{element}={value:+.5f}" for element, value in record["be"].items())
    composition_text = ", ".join(
        f"{element}={value:+.5f}" for element, value in record["composition"].items()
    )
    return f"{label} trial {trial} | BE shift: {be_text}\nComposition shift: {composition_text}"


def select_crps_column(frame: pd.DataFrame) -> str:
    if "crps" in frame.columns:
        return "crps"
    if "crps_scaled" in frame.columns:
        return "crps_scaled"
    raise SystemExit("No CRPS column found in score file")


def load_score_points(score_dir: Path, trial_min: int | None, trial_max: int | None, apply_log: bool, cemc_candidates: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cemc_filename = (
        "best_temperature_by_trial_crps.csv"
        if cemc_candidates == "best-temperature"
        else "crps_by_trial_temperature.csv"
    )
    cemc = pd.read_csv(score_dir / cemc_filename)
    random = pd.read_csv(score_dir / "random_crps_by_trial.csv")
    cemc = filter_trials(cemc, trial_min, trial_max)
    random = filter_trials(random, trial_min, trial_max)

    cemc_crps_col = select_crps_column(cemc)
    random_crps_col = select_crps_column(random)

    for col in ("trial", "temperature", "tau", "mse", cemc_crps_col):
        if col in cemc.columns:
            cemc[col] = pd.to_numeric(cemc[col], errors="coerce")
    for col in ("trial", "tau", "mse", random_crps_col):
        if col in random.columns:
            random[col] = pd.to_numeric(random[col], errors="coerce")

    cemc = cemc.dropna(subset=["trial", "temperature", "tau", "mse", cemc_crps_col]).copy()
    random = random.dropna(subset=["trial", "tau", "mse", random_crps_col]).copy()
    if cemc_crps_col != "crps":
        cemc = cemc.rename(columns={cemc_crps_col: "crps"})
    else:
        cemc["crps"] = pd.to_numeric(cemc["crps"], errors="coerce")
    if random_crps_col != "crps":
        random = random.rename(columns={random_crps_col: "crps"})
    else:
        random["crps"] = pd.to_numeric(random["crps"], errors="coerce")

    cemc = cemc.dropna(subset=["crps"]).copy()
    random = random.dropna(subset=["crps"]).copy()
    cemc["trial"] = cemc["trial"].astype(int)
    random["trial"] = random["trial"].astype(int)
    cemc["method"] = "cemc"
    random["method"] = "random"
    random["temperature"] = np.nan

    if apply_log:
        for frame in (cemc, random):
            for col in ("tau", "mse", "crps"):
                frame[col] = signed_log1p(frame[col].to_numpy(float))
    return cemc.reset_index(drop=True), random.reset_index(drop=True)


def kde_peak_selection(
    points: pd.DataFrame, method_name: str, second_peak_min_mahalanobis: float
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    clean = points.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau", "mse", "crps"]).copy()
    if clean.empty:
        raise SystemExit(f"No metric points available for method {method_name}")
    coords = clean[["tau", "mse", "crps"]].to_numpy(float)
    kde = gaussian_kde(coords.T)
    clean["_kde_density"] = kde(coords.T)
    candidate = clean if (clean["tau"] > 0).sum() == 0 else clean[clean["tau"] > 0]
    rep = candidate.loc[candidate["_kde_density"].idxmax()].copy()
    rep["_method"] = method_name
    delta = candidate[["tau", "mse", "crps"]].to_numpy(float) - rep[["tau", "mse", "crps"]].to_numpy(float)
    inv_cov = np.linalg.pinv(kde.covariance)
    distance = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, inv_cov, delta), 0.0))
    candidate = candidate.copy()
    candidate["_distance_from_first_peak"] = distance
    distinct = candidate[candidate["_distance_from_first_peak"] >= second_peak_min_mahalanobis]
    if distinct.empty:
        distinct = candidate.drop(index=rep.name, errors="ignore")
    if distinct.empty:
        raise SystemExit(f"No second KDE peak candidate available for method {method_name}")
    second = distinct.loc[distinct["_kde_density"].idxmax()].copy()
    second["_method"] = method_name
    clean["_distance_from_first_peak"] = np.nan
    clean.loc[candidate.index, "_distance_from_first_peak"] = candidate["_distance_from_first_peak"]
    return rep, second, clean


def best_score_selection(points: pd.DataFrame) -> pd.Series:
    """Select the global best candidate exactly as documented in best_score_selection.md."""
    clean = points.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tau", "mse", "crps"]
    ).copy()
    if clean.empty:
        raise SystemExit("No CEMC candidate available for global best-score selection")
    for column, higher_is_better in (("tau", True), ("mse", False), ("crps", False)):
        values = clean[column].to_numpy(float)
        low, high = float(values.min()), float(values.max())
        if high - low < 1.0e-14:
            distance = np.zeros_like(values)
        elif higher_is_better:
            distance = (high - values) / (high - low)
        else:
            distance = (values - low) / (high - low)
        clean[f"d_{column}"] = distance
    clean["best_score"] = np.sqrt(
        (clean["d_tau"] ** 2 + clean["d_mse"] ** 2 + clean["d_crps"] ** 2) / 3.0
    )
    return clean.sort_values(
        ["best_score", "tau", "mse", "crps"],
        ascending=[True, False, True, True],
    ).iloc[0].copy()


def plot_histograms3d(
    points_by_method: dict[str, pd.DataFrame], reps: dict[str, pd.Series], output: Path,
    dpi: int, point_size: float, alpha: float, title_prefix: str = "",
    global_best: dict[str, pd.Series] | None = None,
) -> None:
    columns = ("tau", "mse", "crps")
    combined = pd.concat(
        [points_by_method[method].loc[:, columns] for method in ("cemc", "random")],
        ignore_index=True,
    )
    axis_limits: dict[str, tuple[float, float]] = {}
    axis_ticks: dict[str, np.ndarray] = {}
    bin_edges = []
    for column in columns:
        low = float(combined[column].min())
        high = float(combined[column].max())
        if np.isclose(low, high):
            padding = max(abs(low) * 0.05, 0.5)
        else:
            padding = (high - low) * 0.05
        limits = (low - padding, high + padding)
        axis_limits[column] = limits
        axis_ticks[column] = MaxNLocator(nbins=5).tick_values(*limits)
        bin_edges.append(np.linspace(limits[0], limits[1], 16))

    # Assign every point the probability mass of its 3-D histogram bin.
    # Each method is normalized independently, so its bin probabilities sum to 1.
    bin_probabilities: dict[str, np.ndarray] = {}
    for method in ("cemc", "random"):
        values = points_by_method[method].loc[:, columns].to_numpy(float)
        counts, _ = np.histogramdd(values, bins=bin_edges)
        probabilities = counts / counts.sum()
        indices = [
            np.clip(np.searchsorted(edges, values[:, dim], side="right") - 1, 0, len(edges) - 2)
            for dim, edges in enumerate(bin_edges)
        ]
        bin_probabilities[method] = probabilities[tuple(indices)]
    probability_norm = Normalize(
        vmin=0.0,
        vmax=max(float(values.max()) for values in bin_probabilities.values()),
    )

    fig = plt.figure(figsize=(15.5, 6.6))
    for i, method in enumerate(["cemc", "random"], start=1):
        points = points_by_method[method]
        rep = reps[method]
        # Disable mplot3d's depth-based artist reordering so the representative
        # marker is always painted after (and above) the point cloud.
        ax = fig.add_subplot(1, 2, i, projection="3d", computed_zorder=False)
        sc = ax.scatter(
            points["tau"], points["mse"], points["crps"],
            c=bin_probabilities[method], s=point_size, cmap="viridis",
            norm=probability_norm, alpha=alpha,
            zorder=1,
        )
        ax.scatter(
            rep["tau"], rep["mse"], rep["crps"], marker="*", s=240,
            color="red", edgecolors="white", linewidths=1.0,
            depthshade=False, zorder=1000,
            label=f"Selected {'CEMC' if method == 'cemc' else 'Homogeneous'}: trial {int(rep['trial'])}"
        )
        if global_best is not None and method in global_best:
            best = global_best[method]
            temperature_text = (
                f", T={int(best['temperature'])} K" if pd.notna(best.get("temperature", np.nan)) else ""
            )
            ax.scatter(
                best["tau"], best["mse"], best["crps"],
                marker="*", s=285, color="gold", edgecolors="black", linewidths=1.0,
                depthshade=False, zorder=900,
                label=f"Global best score: trial {int(best['trial'])}{temperature_text}",
            )

        if COMPACT:
            title = "CEMC" if method == "cemc" else "Homogeneous"
        elif method == "cemc":
            t = int(rep["temperature"]) if pd.notna(rep["temperature"]) else -1
            title = f"MC/CEMC: trial {int(rep['trial'])}, T={t} K"
        else:
            title = f"Homogeneous: trial {int(rep['trial'])}"

        detail = "" if COMPACT else (
            f"\ntau={float(rep['tau']):.4f}, mse={float(rep['mse']):.4f}, crps={float(rep['crps']):.4f}"
            f"\nKDE density={float(rep['_kde_density']):.4e}"
        )
        ax.set_title(f"{title}{detail}", pad=8, fontsize=18 if COMPACT else None)
        axis_label_size = 14 if COMPACT else None
        ax.set_xlabel("Kendall tau", labelpad=13 if COMPACT else 4, fontsize=axis_label_size)
        ax.set_ylabel("MSE", labelpad=15 if COMPACT else 4, fontsize=axis_label_size)
        ax.set_zlabel("CRPS", labelpad=17 if COMPACT else 4, fontsize=axis_label_size)
        ax.set_xticks(axis_ticks["tau"])
        ax.set_yticks(axis_ticks["mse"])
        ax.set_zticks(axis_ticks["crps"])
        ax.set_xlim(axis_limits["tau"])
        ax.set_ylim(axis_limits["mse"])
        ax.set_zlim(axis_limits["crps"])
        if COMPACT:
            ax.zaxis.label.set_clip_on(False)
        if COMPACT:
            ax.tick_params(axis="x", pad=2)
            ax.tick_params(axis="y", pad=3)
            ax.tick_params(axis="z", pad=5)
        if not COMPACT:
            ax.legend(fontsize=8, frameon=False)

    colorbar = fig.colorbar(
        sc, ax=fig.axes, shrink=0.58 if COMPACT else 0.65,
        pad=0.075 if COMPACT else 0.035, aspect=20,
    )
    colorbar.set_label(
        "Histogram-bin probability",
        labelpad=15 if COMPACT else 4,
    )
    if not COMPACT:
        fig.suptitle(f"{title_prefix}tau/MSE/CRPS 3-D histograms", weight="bold", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, dpi=dpi, bbox_inches="tight",
        pad_inches=0.24 if COMPACT else 0.1,
    )
    plt.close(fig)


def activity_columns(activity_scale: str) -> tuple[str, str, str]:
    if activity_scale == "scaled":
        return "experimental_activity_scaled", "cemc_pred_activity_scaled", "random_pred_activity_scaled"
    if activity_scale == "raw":
        return "experimental_activity", "cemc_pred_activity_mean", "random_pred_activity_mean"
    raise SystemExit(f"Unknown activity scale: {activity_scale}")


def load_activity_for_trial(results_root: Path, trial: int, world_size: int | None) -> pd.DataFrame:
    shards = [results_root / f"shard_{i:02d}" for i in range(world_size)] if world_size is not None else sorted(results_root.glob("shard_*"))
    frames = []
    for shard in shards:
        path = shard / "predicted_activity_selected_by_trial.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df = df.dropna(subset=["trial", "composition_index"])
        df["trial"] = df["trial"].astype(int)
        d = df[df["trial"] == trial].copy()
        if not d.empty:
            d["source_shard"] = shard.name
            frames.append(d)
            break
    if not frames:
        raise SystemExit(f"No predicted activity rows found for trial {trial}")
    out = pd.concat(frames, ignore_index=True)
    out["composition_index"] = pd.to_numeric(out["composition_index"], errors="coerce")
    out = out.dropna(subset=["composition_index"]).copy()
    out["composition_index"] = out["composition_index"].astype(int)
    return out.drop_duplicates("composition_index", keep="last").sort_values("composition_index")


def load_mc_activity_from_zarr(results_root: Path, source_shard: str, trial: int, temperature: int, activity_scale: str) -> np.ndarray:
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
    return np.zeros_like(values) if abs(den) < 1.0e-30 else -(values - np.nanmin(values)) / den


def to_grid_from_trial(df: pd.DataFrame, column: str, n_valid: int, apply_log: bool) -> np.ndarray:
    comp = df["composition_index"].to_numpy(int)
    actual = np.sort(np.unique(comp))
    if len(actual) != n_valid or not np.array_equal(actual, np.arange(n_valid)):
        raise SystemExit(f"Mask/data mismatch: expected composition indices 0..{n_valid - 1}")
    vals = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
    if apply_log:
        vals = signed_log1p(vals)
    out = np.full(n_valid, np.nan, dtype=float)
    out[comp] = vals
    return out


def to_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, np.nan)
    grid[~mask] = values
    return grid


def plot_maps(
    grids: dict[str, np.ndarray], reps: dict[str, pd.Series], output: Path,
    use_log: bool, cmap: str, dpi: int, title_prefix: str = "",
    shift_annotation: str | None = None,
    suptitle: str | None = None,
) -> None:
    # allvals = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    # if allvals.size == 0:
        # vmin, vmax = -1.0, 0.0
    # else:
        # vmin = float(np.percentile(allvals, 2))
        # vmax = float(np.percentile(allvals, 98))
        # if vmin == vmax:
            # vmin -= 1.0
            # vmax += 1.0
    vmin, vmax = -1.0, 0.0
    
    mc = reps["cemc"]
    rs = reps["random"]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
    exp_title = "Experiment"
    mc_title = "CEMC" if COMPACT else (
        f"MC/CEMC: trial {int(mc['trial'])}, T={int(mc['temperature'])} K\n"
        f"tau={float(mc['tau']):.4f}, mse={float(mc['mse']):.4f}, crps={float(mc['crps']):.4f}"
    )
    rs_title = "Homogeneous" if COMPACT else (
        f"Homogeneous: trial {int(rs['trial'])}\n"
        f"tau={float(rs['tau']):.4f}, mse={float(rs['mse']):.4f}, crps={float(rs['crps']):.4f}"
    )
    for ax, key, title in zip(axes, ["experimental", "cemc", "random"], [exp_title, mc_title, rs_title]):
        image = ax.imshow(grids[key], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=28 if COMPACT else 10.0, pad=8)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.05, top=0.91, wspace=0.07)
    cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    cb = fig.colorbar(image, cax=cax, ticks=np.linspace(-1.0, 0.0, 6))
    label = "Scaled ORR activity" if COMPACT else (
        "Activity (log)" if use_log else "Activity (nolog)"
    )
    cb.set_label(label, fontsize=18)
    cb.ax.tick_params(labelsize=18)
    if not COMPACT:
        fig.suptitle(
            suptitle or f"{title_prefix}KDE-selected activity map representatives",
            weight="bold",
        )
    if shift_annotation and not COMPACT:
        fig.text(
            0.5, 0.012, shift_annotation, ha="center", va="bottom",
            fontsize=8.2, family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.95},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_experiment_mc_map(
    experimental: np.ndarray, cemc: np.ndarray, rep: pd.Series, output: Path,
    use_log: bool, cmap: str, dpi: int,
) -> None:
    # allvals = np.concatenate([experimental[~np.isnan(experimental)], cemc[~np.isnan(cemc)]])
    # vmin, vmax = float(np.percentile(allvals, 2)), float(np.percentile(allvals, 98))
    # if vmin == vmax:
        # vmin, vmax = vmin - 1.0, vmax + 1.0
    vmin, vmax = -1.0, 0.0
    
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), constrained_layout=True)
    titles = ["Experiment", (
        f"MC/CEMC: trial {int(rep['trial'])}, T={int(rep['temperature'])} K\n"
        f"tau={float(rep['tau']):.4f}, mse={float(rep['mse']):.4f}, "
        f"crps={float(rep['crps']):.4f}\nideal score={float(rep['_ideal_score']):.4f}"
    )]
    for ax, grid, title in zip(axes, [experimental, cemc], titles):
        image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis("off")
    cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
    cb.set_label("Activity (log)" if use_log else "Activity (nolog)", rotation=270, labelpad=15)
    fig.suptitle("high_tau_low_mse_crps_activity_map", weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global COMPACT
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--world-size", type=int, default=5)
    p.add_argument("--trial-min", type=int, default=None)
    p.add_argument("--trial-max", type=int, default=None)
    p.add_argument("--activity-scale", choices=["scaled", "raw"], default="scaled")
    p.add_argument("--best-score-dir", default=None)
    p.add_argument(
        "--cemc-candidates", choices=["best-temperature", "all-temperatures"],
        default="best-temperature",
        help="Use one preselected best temperature per trial by default",
    )
    p.add_argument("--mask", default="/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy")
    p.add_argument("--experimental-nolog", default="orr_matched_activity.json")
    p.add_argument("--experimental-log", default="orr_log_matched_activity.json")
    p.add_argument("--be-shifts", default="results/shift_metadata/trial_be_shifts.csv")
    p.add_argument("--composition-shifts", default="results/shift_metadata/trial_composition_shifts.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--clip-lower", type=float, default=1.0, help="Lower activity clipping percentile")
    p.add_argument("--clip-upper", type=float, default=99.0, help="Upper activity clipping percentile")
    p.add_argument("--cmap", default="viridis_r")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--point-size", type=float, default=7.0, help="3D scatter marker area")
    p.add_argument("--alpha", type=float, default=0.22, help="3D scatter opacity (lower is more transparent)")
    p.add_argument("--compact", action="store_true", help="Use larger, minimal text for composite figures")
    p.add_argument(
        "--second-peak-min-mahalanobis", type=float, default=1.0,
        help="Minimum KDE-covariance Mahalanobis distance from the first peak (default: 1.0)",
    )
    p.add_argument(
        "--log-transform", action=argparse.BooleanOptionalAction, default=True,
        help="Use log-domain activities (default); pass --no-log-transform for legacy nolog mode",
    )
    args = p.parse_args()
    COMPACT = args.compact
    if COMPACT:
        plt.rcParams.update({"font.size": 15, "axes.labelsize": 16,
                             "xtick.labelsize": 13, "ytick.labelsize": 13})
    if not (0.0 <= args.clip_lower < args.clip_upper <= 100.0):
        p.error("clipping percentiles must satisfy 0 <= lower < upper <= 100")

    results_root = Path(args.results_root).resolve()
    activity_domain = "log" if args.log_transform else "nolog"
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else (
        results_root / f"crps_selection_activity_{activity_domain}"
    )
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (
        results_root / f"modal_tau_mse_crps_activity_map_kde_clipped_{args.activity_scale}_{'log' if args.log_transform else 'nolog'}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # These score tables are computed after the selected activity-domain
    # transform and independent [-1, 0] scaling of experiment/prediction.
    cemc, random = load_score_points(
        score_dir, args.trial_min, args.trial_max, apply_log=False,
        cemc_candidates=args.cemc_candidates,
    )
    all_temperature_cemc, _ = load_score_points(
        score_dir, args.trial_min, args.trial_max, apply_log=False,
        cemc_candidates="all-temperatures",
    )

    reps: dict[str, pd.Series] = {}
    second_reps: dict[str, pd.Series] = {}
    assigned_frames = []
    for method, frame in (("cemc", cemc), ("random", random)):
        rep, second_rep, assigned = kde_peak_selection(
            frame, method, args.second_peak_min_mahalanobis
        )
        reps[method] = rep
        second_reps[method] = second_rep
        assigned_frames.append(assigned)
    global_best_mc = best_score_selection(all_temperature_cemc)
    global_best_random = best_score_selection(random)
    shifts = load_trial_shifts(Path(args.be_shifts), Path(args.composition_shifts))

    hist_png = out_dir / "tau_mse_crps_3d_kde_histogram.png"
    mc_trial = int(reps["cemc"]["trial"])
    rs_trial = int(reps["random"]["trial"])
    map_png = out_dir / f"mc_trial_{mc_trial:04d}_random_trial_{rs_trial:04d}_kde_activity_maps.png"
    rep_csv = out_dir / "modal_tau_mse_crps_representative_trials_kde.csv"
    assignment_csv = out_dir / "tau_mse_crps_point_kde_assignments.csv"
    mc_values_csv = out_dir / f"trial_{mc_trial:04d}_mc_activity_map_values.csv"
    rs_values_csv = out_dir / f"trial_{rs_trial:04d}_random_activity_map_values.csv"

    point_frames = {"cemc": assigned_frames[0], "random": assigned_frames[1]}
    plot_histograms3d(
        point_frames, reps, hist_png, args.dpi, args.point_size, args.alpha,
        global_best={"cemc": global_best_mc, "random": global_best_random},
    )
    second_hist_png = out_dir / "second_tau_mse_crps_3d_kde_histogram.png"
    plot_histograms3d(
        point_frames, second_reps, second_hist_png, args.dpi, args.point_size, args.alpha,
        title_prefix="second_",
    )

    mask = np.load(args.mask).astype(bool)
    n_valid = int(np.count_nonzero(~mask))

    mc = reps["cemc"]
    rs = reps["random"]
    mc_act = load_activity_for_trial(results_root, int(mc["trial"]), args.world_size)
    rs_act = load_activity_for_trial(results_root, int(rs["trial"]), args.world_size)
    mc_source_shard = str(mc_act["source_shard"].iloc[0])
    rs_source_shard = str(rs_act["source_shard"].iloc[0])

    experimental_path = Path(args.experimental_log if args.log_transform else args.experimental_nolog).resolve()
    exp_vals = load_matched_activity(experimental_path, n_valid)

    # Stored prediction means are log activities. For the legacy nolog view,
    # exponentiate those means, then apply the same [-1, 0] scaling as log mode.
    cemc_log_vals_raw = load_mc_activity_from_zarr(
        results_root, mc_source_shard, int(mc["trial"]), int(mc["temperature"]), "raw"
    )
    random_log_vals_raw = to_grid_from_trial(
        rs_act, "random_pred_activity_mean", n_valid, apply_log=False
    )
    if args.log_transform:
        cemc_unscaled_raw = cemc_log_vals_raw
        random_unscaled_raw = random_log_vals_raw
        exp_col = "experimental_log_scaled_activity"
        random_col = "random_log_scaled_activity"
    else:
        cemc_unscaled_raw = np.exp(cemc_log_vals_raw)
        random_unscaled_raw = np.exp(random_log_vals_raw)
        exp_col = "experimental_nolog_scaled_activity"
        random_col = "random_nolog_scaled_activity"
    
    cemc_unscaled_clipped = clip_outliers(cemc_unscaled_raw, args.clip_lower, args.clip_upper)
    random_unscaled_clipped = clip_outliers(random_unscaled_raw, args.clip_lower, args.clip_upper)
    
    cemc_vals = scale_prediction_to_minus1_0(cemc_unscaled_clipped)
    rs_vals = scale_prediction_to_minus1_0(random_unscaled_clipped)

    grids = {
        "experimental": to_grid(exp_vals, mask),
        "cemc": to_grid(cemc_vals, mask),
        "random": to_grid(rs_vals, mask),
    }

    primary_shift_annotation = format_trial_shifts("MC/CEMC", int(mc["trial"]), shifts)
    plot_maps(
        grids, reps, map_png, args.log_transform, args.cmap, args.dpi,
        shift_annotation=primary_shift_annotation,
    )

    rep_record = {
        "mode": "log" if args.log_transform else "nolog",
        "activity_scale": args.activity_scale,
        "clip_lower_percentile": args.clip_lower,
        "clip_upper_percentile": args.clip_upper,
        "mc_trial": int(mc["trial"]),
        "mc_temperature": int(mc["temperature"]),
        "random_trial": int(rs["trial"]),
        "mc_tau": float(mc["tau"]),
        "mc_mse": float(mc["mse"]),
        "mc_crps": float(mc["crps"]),
        "random_tau": float(rs["tau"]),
        "random_mse": float(rs["mse"]),
        "random_crps": float(rs["crps"]),
        "mc_source_shard": mc_source_shard,
        "random_source_shard": rs_source_shard,
        "mc_kde_density": float(mc["_kde_density"]),
        "random_kde_density": float(rs["_kde_density"]),
    }
    pd.DataFrame([rep_record]).to_csv(rep_csv, index=False)
    pd.concat(assigned_frames, ignore_index=True).to_csv(assignment_csv, index=False)

    pd.DataFrame({"composition_index": np.arange(len(exp_vals)), exp_col: exp_vals}).to_csv(mc_values_csv, index=False)
    pd.DataFrame({"composition_index": np.arange(len(rs_vals)), random_col: rs_vals}).to_csv(rs_values_csv, index=False)

    np.save(out_dir / f"trial_{mc_trial:04d}_cemc_grid.npy", grids["cemc"])
    np.save(out_dir / f"trial_{rs_trial:04d}_random_grid.npy", grids["random"])

    second_mc = second_reps["cemc"]
    second_rs = second_reps["random"]
    second_mc_trial = int(second_mc["trial"])
    second_rs_trial = int(second_rs["trial"])
    second_mc_act = load_activity_for_trial(results_root, second_mc_trial, args.world_size)
    second_rs_act = load_activity_for_trial(results_root, second_rs_trial, args.world_size)
    second_mc_shard = str(second_mc_act["source_shard"].iloc[0])
    second_rs_shard = str(second_rs_act["source_shard"].iloc[0])
    second_cemc_log = load_mc_activity_from_zarr(
        results_root, second_mc_shard, second_mc_trial, int(second_mc["temperature"]), "raw"
    )
    second_random_log = to_grid_from_trial(
        second_rs_act, "random_pred_activity_mean", n_valid, apply_log=False
    )
    if args.log_transform:
        second_cemc_unscaled_raw = second_cemc_log
        second_random_unscaled_raw = second_random_log
    else:
        second_cemc_unscaled_raw = np.exp(second_cemc_log)
        second_random_unscaled_raw = np.exp(second_random_log)
    
    second_cemc_unscaled_clipped = clip_outliers(second_cemc_unscaled_raw, args.clip_lower, args.clip_upper)
    second_random_unscaled_clipped = clip_outliers(second_random_unscaled_raw, args.clip_lower, args.clip_upper)
    
    second_cemc_vals = scale_prediction_to_minus1_0(second_cemc_unscaled_clipped)
    second_random_vals = scale_prediction_to_minus1_0(second_random_unscaled_clipped)
    second_grids = {
        "experimental": to_grid(exp_vals, mask),
        "cemc": to_grid(second_cemc_vals, mask),
        "random": to_grid(second_random_vals, mask),
    }
    second_map_png = out_dir / (
        f"second_mc_trial_{second_mc_trial:04d}_random_trial_{second_rs_trial:04d}_kde_activity_maps.png"
    )
    plot_maps(
        second_grids, second_reps, second_map_png, args.log_transform, args.cmap, args.dpi,
        title_prefix="second_",
        shift_annotation=format_trial_shifts("Second MC/CEMC", second_mc_trial, shifts),
    )
    second_rep_csv = out_dir / "second_modal_tau_mse_crps_representative_trials_kde.csv"
    second_records = []
    for method, selected in second_reps.items():
        second_records.append({
            "mode": "log" if args.log_transform else "nolog",
            "clip_lower_percentile": args.clip_lower,
            "clip_upper_percentile": args.clip_upper,
            "method": method,
            "trial": int(selected["trial"]),
            "temperature": int(selected["temperature"]) if pd.notna(selected["temperature"]) else np.nan,
            "tau": float(selected["tau"]), "mse": float(selected["mse"]),
            "crps": float(selected["crps"]), "kde_density": float(selected["_kde_density"]),
            "distance_from_first_peak": float(selected["_distance_from_first_peak"]),
            "second_peak_min_mahalanobis": args.second_peak_min_mahalanobis,
        })
    pd.DataFrame(second_records).to_csv(second_rep_csv, index=False)
    pd.DataFrame({"composition_index": np.arange(n_valid), exp_col: exp_vals,
                  "second_mc_activity": second_cemc_vals}).to_csv(
        out_dir / f"second_trial_{second_mc_trial:04d}_mc_activity_map_values.csv", index=False
    )
    pd.DataFrame({"composition_index": np.arange(n_valid),
                  "second_random_activity": second_random_vals}).to_csv(
        out_dir / f"second_trial_{second_rs_trial:04d}_random_activity_map_values.csv", index=False
    )
    np.save(out_dir / f"second_trial_{second_mc_trial:04d}_cemc_grid.npy", second_grids["cemc"])
    np.save(out_dir / f"second_trial_{second_rs_trial:04d}_random_grid.npy", second_grids["random"])

    print(
        f"selected MC: trial={mc_trial} T={int(mc['temperature'])} K "
        f"tau={float(mc['tau']):.6f}, mse={float(mc['mse']):.6f}, crps={float(mc['crps']):.6f}"
    )
    print(primary_shift_annotation)
    print(
        f"selected RS: trial={rs_trial} "
        f"tau={float(rs['tau']):.6f}, mse={float(rs['mse']):.6f}, crps={float(rs['crps']):.6f}"
    )
    print(f"wrote {hist_png}")
    print(f"wrote {map_png}")
    print(f"wrote {rep_csv}")
    print(f"wrote {assignment_csv}")
    print(f"wrote {mc_values_csv}")
    print(f"wrote {rs_values_csv}")
    print(
        f"selected second MC: trial={second_mc_trial} T={int(second_mc['temperature'])} K; "
        f"distance={float(second_mc['_distance_from_first_peak']):.4f}"
    )
    print(
        f"selected second RS: trial={second_rs_trial}; "
        f"distance={float(second_rs['_distance_from_first_peak']):.4f}"
    )
    print(f"wrote {second_hist_png}")
    print(f"wrote {second_map_png}")
    print(f"wrote {second_rep_csv}")

    high_trial = int(global_best_mc["trial"])
    high_temp = int(global_best_mc["temperature"])
    high_act = load_activity_for_trial(results_root, high_trial, args.world_size)
    high_shard = str(high_act["source_shard"].iloc[0])
    high_log_values = load_mc_activity_from_zarr(
        results_root, high_shard, high_trial, high_temp, "raw"
    )
    high_unscaled_raw = high_log_values if args.log_transform else np.exp(high_log_values)
    high_unscaled_clipped = clip_outliers(high_unscaled_raw, args.clip_lower, args.clip_upper)
    high_values = scale_prediction_to_minus1_0(high_unscaled_clipped)
    high_grid = to_grid(high_values, mask)
    best_random_trial = int(global_best_random["trial"])
    best_random_act = load_activity_for_trial(results_root, best_random_trial, args.world_size)
    high_random_log = to_grid_from_trial(
        best_random_act, "random_pred_activity_mean", n_valid, apply_log=False
    )
    high_random_unscaled = high_random_log if args.log_transform else np.exp(high_random_log)
    high_random_clipped = clip_outliers(high_random_unscaled, args.clip_lower, args.clip_upper)
    high_random_values = scale_prediction_to_minus1_0(high_random_clipped)
    high_random_grid = to_grid(high_random_values, mask)
    high_random_rep = global_best_random
    high_map_png = out_dir / (
        f"global_best_cemc_trial_{high_trial:04d}_T_{high_temp:04d}_"
        f"random_trial_{best_random_trial:04d}_activity_maps.png"
    )
    plot_maps(
        {"experimental": grids["experimental"], "cemc": high_grid, "random": high_random_grid},
        {"cemc": global_best_mc, "random": high_random_rep},
        high_map_png, args.log_transform, args.cmap, args.dpi,
        shift_annotation=format_trial_shifts("Global best MC/CEMC", high_trial, shifts),
        suptitle="Global best-score trial activity maps",
    )
    high_record = {
        "mode": "log" if args.log_transform else "nolog",
        "clip_lower_percentile": args.clip_lower,
        "clip_upper_percentile": args.clip_upper,
        "trial": high_trial, "temperature": high_temp,
        "tau": float(global_best_mc["tau"]), "mse": float(global_best_mc["mse"]),
        "crps": float(global_best_mc["crps"]),
        "d_tau": float(global_best_mc["d_tau"]),
        "d_mse": float(global_best_mc["d_mse"]),
        "d_crps": float(global_best_mc["d_crps"]),
        "best_score": float(global_best_mc["best_score"]),
        "candidate_scope": "all_trial_temperature_pairs",
        "source_shard": high_shard,
    }
    random_best_record = {
        "mode": "log" if args.log_transform else "nolog",
        "clip_lower_percentile": args.clip_lower,
        "clip_upper_percentile": args.clip_upper,
        "trial": best_random_trial,
        "tau": float(global_best_random["tau"]),
        "mse": float(global_best_random["mse"]),
        "crps": float(global_best_random["crps"]),
        "d_tau": float(global_best_random["d_tau"]),
        "d_mse": float(global_best_random["d_mse"]),
        "d_crps": float(global_best_random["d_crps"]),
        "best_score": float(global_best_random["best_score"]),
        "candidate_scope": "all_random_trials",
        "source_shard": str(best_random_act["source_shard"].iloc[0]),
    }
    high_csv = out_dir / "high_tau_low_mse_crps_representative_trial.csv"
    pd.DataFrame([high_record]).to_csv(high_csv, index=False)
    global_best_csv = out_dir / "global_best_score_cemc_random.csv"
    pd.DataFrame([
        {"method": "cemc", **high_record},
        {"method": "random", **random_best_record},
    ]).to_csv(global_best_csv, index=False)
    pd.DataFrame({"composition_index": np.arange(n_valid),
                  "experimental_activity": exp_vals,
                  "high_tau_low_mse_crps_mc_activity": high_values,
                  "same_trial_random_activity": high_random_values}).to_csv(
        out_dir / f"high_tau_low_mse_crps_trial_{high_trial:04d}_activity_values.csv", index=False
    )
    np.save(out_dir / f"high_tau_low_mse_crps_trial_{high_trial:04d}_grid.npy", high_grid)
    np.save(out_dir / f"global_best_random_trial_{best_random_trial:04d}_grid.npy", high_random_grid)
    print(
        f"selected global best-score MC from all trial/temperature pairs: "
        f"trial={high_trial} T={high_temp} K "
        f"tau={float(global_best_mc['tau']):.6f}, mse={float(global_best_mc['mse']):.6f}, "
        f"crps={float(global_best_mc['crps']):.6f}, best_score={float(global_best_mc['best_score']):.6f}"
    )
    print(format_trial_shifts("Global best MC/CEMC", high_trial, shifts))
    print(
        f"selected global best-score Homogeneous: trial={best_random_trial} "
        f"tau={float(global_best_random['tau']):.6f}, mse={float(global_best_random['mse']):.6f}, "
        f"crps={float(global_best_random['crps']):.6f}, "
        f"best_score={float(global_best_random['best_score']):.6f}"
    )
    print(f"wrote {high_map_png}")
    print(f"wrote {high_csv}")
    print(f"wrote {global_best_csv}")


if __name__ == "__main__":
    main()
