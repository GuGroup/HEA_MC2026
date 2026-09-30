#!/usr/bin/env python3
"""KDE-smoothed 3D tau/MSE/CRPS selection and activity-map plotting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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


def high_tau_low_errors_selection(points: pd.DataFrame) -> pd.Series:
    """Select the observed tau>0 point nearest the ideal tau/max, errors/min corner."""
    clean = points.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tau", "mse", "crps"]
    ).copy()
    candidate = clean[clean["tau"] > 0].copy()
    if candidate.empty:
        raise SystemExit("No tau > 0 CEMC candidate for high-tau/low-error selection")
    for column, higher_is_better in (("tau", True), ("mse", False), ("crps", False)):
        values = candidate[column].to_numpy(float)
        low, high = float(values.min()), float(values.max())
        if high - low < 1.0e-14:
            distance = np.zeros_like(values)
        elif higher_is_better:
            distance = (high - values) / (high - low)
        else:
            distance = (values - low) / (high - low)
        candidate[f"_ideal_d_{column}"] = distance
    candidate["_ideal_score"] = np.sqrt(
        (candidate["_ideal_d_tau"] ** 2 + candidate["_ideal_d_mse"] ** 2
         + candidate["_ideal_d_crps"] ** 2) / 3.0
    )
    return candidate.sort_values(
        ["_ideal_score", "tau", "mse", "crps"],
        ascending=[True, False, True, True],
    ).iloc[0].copy()


def plot_histograms3d(points_by_method: dict[str, pd.DataFrame], reps: dict[str, pd.Series], output: Path, dpi: int, point_size: float, alpha: float, title_prefix: str = "") -> None:
    fig = plt.figure(figsize=(15.0, 6.8))
    for i, method in enumerate(["cemc", "random"], start=1):
        points = points_by_method[method]
        rep = reps[method]
        ax = fig.add_subplot(1, 2, i, projection="3d")
        sc = ax.scatter(
            points["tau"], points["mse"], points["crps"],
            c=points["_kde_density"], s=point_size, cmap="viridis", alpha=alpha
        )
        ax.scatter(
            rep["tau"], rep["mse"], rep["crps"], marker="X", s=170,
            color="red", edgecolors="white", linewidths=1.0,
            label=f"Selected {method}: trial {int(rep['trial'])}"
        )

        if method == "cemc":
            t = int(rep["temperature"]) if pd.notna(rep["temperature"]) else -1
            title = f"MC/CEMC: trial {int(rep['trial'])}, T={t} K"
        else:
            title = f"Random/RS: trial {int(rep['trial'])}"

        ax.set_title(
            f"{title}\n"
            f"tau={float(rep['tau']):.4f}, mse={float(rep['mse']):.4f}, crps={float(rep['crps']):.4f}\n"
            f"KDE density={float(rep['_kde_density']):.4e}",
            pad=8
        )
        ax.set_xlabel("Kendall tau")
        ax.set_ylabel("MSE")
        ax.set_zlabel("CRPS")
        ax.legend(fontsize=8, frameon=False)

    fig.colorbar(sc, ax=fig.axes, shrink=0.65, pad=0.035, label="KDE density")
    fig.suptitle(f"{title_prefix}KDE-smoothed tau/MSE/CRPS point clouds", weight="bold", fontsize=15)
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


def plot_maps(grids: dict[str, np.ndarray], reps: dict[str, pd.Series], output: Path, use_log: bool, cmap: str, dpi: int, title_prefix: str = "") -> None:
    allvals = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    if allvals.size == 0:
        vmin, vmax = -1.0, 0.0
    else:
        vmin = float(np.percentile(allvals, 2))
        vmax = float(np.percentile(allvals, 98))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

    mc = reps["cemc"]
    rs = reps["random"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    exp_title = "Experiment"
    mc_title = (
        f"MC/CEMC: trial {int(mc['trial'])}, T={int(mc['temperature'])} K\n"
        f"tau={float(mc['tau']):.4f}, mse={float(mc['mse']):.4f}, crps={float(mc['crps']):.4f}"
    )
    rs_title = (
        f"Random/RS: trial {int(rs['trial'])}\n"
        f"tau={float(rs['tau']):.4f}, mse={float(rs['mse']):.4f}, crps={float(rs['crps']):.4f}"
    )
    for ax, key, title in zip(axes, ["experimental", "cemc", "random"], [exp_title, mc_title, rs_title]):
        image = ax.imshow(grids[key], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10.0, pad=8)
        ax.axis("off")

    cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    label = "Activity"
    label += " (log)" if use_log else " (nolog)"
    cb.set_label(label, rotation=270, labelpad=15)
    fig.suptitle(f"{title_prefix}KDE-selected activity map representatives", weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_mc_map(
    experimental: np.ndarray, cemc: np.ndarray, rep: pd.Series, output: Path,
    use_log: bool, cmap: str, dpi: int,
) -> None:
    allvals = np.concatenate([experimental[~np.isnan(experimental)], cemc[~np.isnan(cemc)]])
    vmin, vmax = float(np.percentile(allvals, 2)), float(np.percentile(allvals, 98))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
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
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cmap", default="bwr_r")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--point-size", type=float, default=7.0, help="3D scatter marker area")
    p.add_argument("--alpha", type=float, default=0.22, help="3D scatter opacity (lower is more transparent)")
    p.add_argument(
        "--second-peak-min-mahalanobis", type=float, default=1.0,
        help="Minimum KDE-covariance Mahalanobis distance from the first peak (default: 1.0)",
    )
    p.add_argument(
        "--log-transform", action="store_true",
        help="Use log-domain activities scaled to [-1, 0] and their precomputed metrics",
    )
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    activity_domain = "log" if args.log_transform else "nolog"
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else (
        results_root / f"crps_selection_activity_{activity_domain}"
    )
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (
        results_root / f"modal_tau_mse_crps_activity_map_kde_{args.activity_scale}_{'log' if args.log_transform else 'nolog'}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # These score tables are computed after the selected activity-domain
    # transform and independent [-1, 0] scaling of experiment/prediction.
    cemc, random = load_score_points(
        score_dir, args.trial_min, args.trial_max, apply_log=False,
        cemc_candidates=args.cemc_candidates,
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
    high_quality_mc = high_tau_low_errors_selection(cemc)

    hist_png = out_dir / "tau_mse_crps_3d_kde_histogram.png"
    mc_trial = int(reps["cemc"]["trial"])
    rs_trial = int(reps["random"]["trial"])
    map_png = out_dir / f"mc_trial_{mc_trial:04d}_random_trial_{rs_trial:04d}_kde_activity_maps.png"
    rep_csv = out_dir / "modal_tau_mse_crps_representative_trials_kde.csv"
    assignment_csv = out_dir / "tau_mse_crps_point_kde_assignments.csv"
    mc_values_csv = out_dir / f"trial_{mc_trial:04d}_mc_activity_map_values.csv"
    rs_values_csv = out_dir / f"trial_{rs_trial:04d}_random_activity_map_values.csv"

    point_frames = {"cemc": assigned_frames[0], "random": assigned_frames[1]}
    plot_histograms3d(point_frames, reps, hist_png, args.dpi, args.point_size, args.alpha)
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
    cemc_log_vals = load_mc_activity_from_zarr(
        results_root, mc_source_shard, int(mc["trial"]), int(mc["temperature"]), "raw"
    )
    random_log_vals = to_grid_from_trial(
        rs_act, "random_pred_activity_mean", n_valid, apply_log=False
    )
    if args.log_transform:
        cemc_unscaled = cemc_log_vals
        random_unscaled = random_log_vals
        exp_col = "experimental_log_scaled_activity"
        random_col = "random_log_scaled_activity"
    else:
        cemc_unscaled = np.exp(cemc_log_vals)
        random_unscaled = np.exp(random_log_vals)
        exp_col = "experimental_nolog_scaled_activity"
        random_col = "random_nolog_scaled_activity"
    cemc_vals = scale_prediction_to_minus1_0(cemc_unscaled)
    rs_vals = scale_prediction_to_minus1_0(random_unscaled)

    grids = {
        "experimental": to_grid(exp_vals, mask),
        "cemc": to_grid(cemc_vals, mask),
        "random": to_grid(rs_vals, mask),
    }

    plot_maps(grids, reps, map_png, args.log_transform, args.cmap, args.dpi)

    rep_record = {
        "mode": "log" if args.log_transform else "nolog",
        "activity_scale": args.activity_scale,
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
        second_cemc_unscaled = second_cemc_log
        second_random_unscaled = second_random_log
    else:
        second_cemc_unscaled = np.exp(second_cemc_log)
        second_random_unscaled = np.exp(second_random_log)
    second_cemc_vals = scale_prediction_to_minus1_0(second_cemc_unscaled)
    second_random_vals = scale_prediction_to_minus1_0(second_random_unscaled)
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
    )
    second_rep_csv = out_dir / "second_modal_tau_mse_crps_representative_trials_kde.csv"
    second_records = []
    for method, selected in second_reps.items():
        second_records.append({
            "mode": "log" if args.log_transform else "nolog",
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

    high_trial = int(high_quality_mc["trial"])
    high_temp = int(high_quality_mc["temperature"])
    high_act = load_activity_for_trial(results_root, high_trial, args.world_size)
    high_shard = str(high_act["source_shard"].iloc[0])
    high_log_values = load_mc_activity_from_zarr(
        results_root, high_shard, high_trial, high_temp, "raw"
    )
    high_unscaled = high_log_values if args.log_transform else np.exp(high_log_values)
    high_values = scale_prediction_to_minus1_0(high_unscaled)
    high_grid = to_grid(high_values, mask)
    high_map_png = out_dir / (
        f"high_tau_low_mse_crps_mc_trial_{high_trial:04d}_T_{high_temp:04d}_activity_map.png"
    )
    plot_experiment_mc_map(
        grids["experimental"], high_grid, high_quality_mc, high_map_png,
        args.log_transform, args.cmap, args.dpi,
    )
    high_record = {
        "mode": "log" if args.log_transform else "nolog",
        "trial": high_trial, "temperature": high_temp,
        "tau": float(high_quality_mc["tau"]), "mse": float(high_quality_mc["mse"]),
        "crps": float(high_quality_mc["crps"]),
        "ideal_d_tau": float(high_quality_mc["_ideal_d_tau"]),
        "ideal_d_mse": float(high_quality_mc["_ideal_d_mse"]),
        "ideal_d_crps": float(high_quality_mc["_ideal_d_crps"]),
        "ideal_score": float(high_quality_mc["_ideal_score"]),
        "source_shard": high_shard,
    }
    high_csv = out_dir / "high_tau_low_mse_crps_representative_trial.csv"
    pd.DataFrame([high_record]).to_csv(high_csv, index=False)
    pd.DataFrame({"composition_index": np.arange(n_valid),
                  "high_tau_low_mse_crps_mc_activity": high_values}).to_csv(
        out_dir / f"high_tau_low_mse_crps_trial_{high_trial:04d}_activity_values.csv", index=False
    )
    np.save(out_dir / f"high_tau_low_mse_crps_trial_{high_trial:04d}_grid.npy", high_grid)
    print(
        f"selected high-tau/low-error MC: trial={high_trial} T={high_temp} K "
        f"tau={float(high_quality_mc['tau']):.6f}, mse={float(high_quality_mc['mse']):.6f}, "
        f"crps={float(high_quality_mc['crps']):.6f}, ideal_score={float(high_quality_mc['_ideal_score']):.6f}"
    )
    print(f"wrote {high_map_png}")
    print(f"wrote {high_csv}")


if __name__ == "__main__":
    main()
