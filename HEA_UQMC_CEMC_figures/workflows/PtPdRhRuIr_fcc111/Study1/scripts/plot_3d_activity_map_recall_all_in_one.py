#!/usr/bin/env python3
"""Generate separate 3-D, activity-map, and recall figures for two selections.

Selections are the positive-tau KDE maximum (``high_probability``) and the
global minimum normalized distance to ideal tau/MSE/CRPS (``best_score``).
This driver reuses the two project plotting scripts so all loading, clipping,
scaling, and recall conventions remain identical.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import plot_3dhistogram_activity_map_clipping as activity_plot  # noqa: E402


def run_checked(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def number_tag(value: float) -> str:
    """Make a filesystem-friendly clipping-percentile label."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def add_density(row, points):
    selected = row.copy()
    xyz = points[["tau", "mse", "crps"]].to_numpy(float)
    selected["_kde_density"] = float(gaussian_kde(xyz.T)(
        selected[["tau", "mse", "crps"]].to_numpy(float)[:, None]
    )[0])
    return selected


def make_separate_histograms(args, score_dir: Path, output_dir: Path) -> tuple[dict, dict]:
    cemc, random = activity_plot.load_score_points(
        score_dir, args.trial_min, args.trial_max, False, args.cemc_candidates
    )
    all_cemc, _ = activity_plot.load_score_points(
        score_dir, args.trial_min, args.trial_max, False, "all-temperatures"
    )
    high_probability, point_frames = {}, {}
    for method, frame in (("cemc", cemc), ("random", random)):
        representative, _, assigned = activity_plot.kde_peak_selection(
            frame, method, args.second_peak_min_mahalanobis
        )
        high_probability[method] = representative
        point_frames[method] = assigned
    best_score = {
        # Selection uses every trial/temperature pair.  Density/coloring uses
        # the trial-level point cloud to avoid an O(N^2) KDE over all pairs.
        "cemc": add_density(activity_plot.best_score_selection(all_cemc), cemc),
        "random": add_density(activity_plot.best_score_selection(random), random),
    }
    high_hist = output_dir / "high_probability_tau_mse_crps_3d_histogram.png"
    best_hist = output_dir / "best_score_tau_mse_crps_3d_histogram.png"
    activity_plot.plot_histograms3d(
        point_frames, high_probability, high_hist, args.dpi, args.point_size,
        args.alpha, title_prefix="High-probability ",
    )
    activity_plot.plot_histograms3d(
        point_frames, best_score, best_hist, args.dpi, args.point_size,
        args.alpha, title_prefix="Global best-score ",
    )
    shutil.copy2(high_hist, output_dir / "tau_mse_crps_3d_kde_histogram.png")
    return high_probability, best_score


def stable_map_names(output_dir: Path, high_probability: dict, best_score: dict) -> None:
    hp = output_dir / (
        f"mc_trial_{int(high_probability['cemc']['trial']):04d}_"
        f"random_trial_{int(high_probability['random']['trial']):04d}_kde_activity_maps.png"
    )
    best = output_dir / (
        f"global_best_cemc_trial_{int(best_score['cemc']['trial']):04d}_"
        f"T_{int(best_score['cemc']['temperature']):04d}_"
        f"random_trial_{int(best_score['random']['trial']):04d}_activity_maps.png"
    )
    if not hp.exists() or not best.exists():
        raise FileNotFoundError(f"Expected activity maps were not generated: {hp}, {best}")
    shutil.copy2(hp, output_dir / "high_probability_activity_maps.png")
    shutil.copy2(best, output_dir / "best_score_activity_maps.png")


def compose_all_in_one(histogram: Path, recall: Path, activity_maps: Path,
                       output: Path, dpi: int) -> None:
    """Arrange the three independently generated figures without panel labels."""
    inputs = (histogram, recall, activity_maps)
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing component figure(s): " + ", ".join(missing))
    images = [plt.imread(path) for path in inputs]
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.0, 1.12),
        left=0.012, right=0.992, bottom=0.015, top=0.99,
        wspace=0.065, hspace=0.07,
    )
    axes = [
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, :]),
    ]
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--world-size", type=int, default=5)
    p.add_argument("--trial-min", type=int, default=None)
    p.add_argument("--trial-max", type=int, default=None)
    p.add_argument("--activity-scale", choices=("scaled", "raw"), default="scaled")
    p.add_argument("--best-score-dir", type=Path, default=None)
    p.add_argument("--cemc-candidates", choices=("best-temperature", "all-temperatures"), default="best-temperature")
    p.add_argument("--mask", type=Path, default=Path("/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy"))
    p.add_argument("--experimental-nolog", type=Path, default=Path("orr_matched_activity.json"))
    p.add_argument("--experimental-log", type=Path, default=Path("orr_log_matched_activity.json"))
    p.add_argument("--be-shifts", type=Path, default=Path("results/shift_metadata/trial_be_shifts.csv"))
    p.add_argument("--composition-shifts", type=Path, default=Path("results/shift_metadata/trial_composition_shifts.csv"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--recall-output-dir", type=Path, default=Path("results/activity_recall_ranking_log"))
    p.add_argument("--run-name", default=None,
                   help="Output subfolder name (default: clip values plus current timestamp)")
    p.add_argument("--clip-lower", type=float, default=1.0)
    p.add_argument("--clip-upper", type=float, default=99.0)
    p.add_argument("--cmap", default="viridis_r")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--point-size", type=float, default=7.0)
    p.add_argument("--alpha", type=float, default=0.22)
    p.add_argument("--second-peak-min-mahalanobis", type=float, default=1.0)
    p.add_argument("--max-fraction", type=float, default=0.60)
    p.add_argument("--n-points", type=int, default=60)
    p.add_argument("--log-transform", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if not (0 <= args.clip_lower < args.clip_upper <= 100):
        p.error("clipping percentiles must satisfy 0 <= lower < upper <= 100")
    activity_plot.COMPACT = True
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13})

    domain = "log" if args.log_transform else "nolog"
    results_root = project_path(args.results_root)
    score_dir = project_path(args.best_score_dir) if args.best_score_dir else results_root / f"crps_selection_activity_{domain}_center_cross_masked"
    output_base = project_path(args.output_dir) if args.output_dir else results_root / f"modal_tau_mse_crps_activity_map_kde_clipped_{args.activity_scale}_{domain}"
    recall_base = project_path(args.recall_output_dir)
    run_name = args.run_name or (
        f"clip_{number_tag(args.clip_lower)}_{number_tag(args.clip_upper)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_base / run_name
    recall_output_dir = recall_base / run_name
    if output_dir.exists() or recall_output_dir.exists():
        p.error(
            f"run output already exists: {run_name!r}; choose a different --run-name "
            "to prevent overwriting"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_transform and args.best_score_dir is None:
        scoring_cmd = [sys.executable, str(SCRIPT_DIR / "score_log_activity_zarr.py"),
            "--results-root", str(results_root), "--world-size", str(args.world_size),
            "--experimental", str(args.experimental_log), "--mask", str(args.mask),
            "--cross-index", "25", "--output-dir", str(score_dir)]
        if args.trial_min is not None:
            scoring_cmd += ["--trial-min", str(args.trial_min)]
        if args.trial_max is not None:
            scoring_cmd += ["--trial-max", str(args.trial_max)]
        run_checked(scoring_cmd)

    clipping_cmd = [sys.executable, str(SCRIPT_DIR / "plot_3dhistogram_activity_map_clipping.py"),
        "--results-root", str(results_root), "--world-size", str(args.world_size),
        "--activity-scale", args.activity_scale, "--best-score-dir", str(score_dir),
        "--cemc-candidates", args.cemc_candidates, "--mask", str(args.mask),
        "--experimental-nolog", str(args.experimental_nolog), "--experimental-log", str(args.experimental_log),
        "--be-shifts", str(args.be_shifts), "--composition-shifts", str(args.composition_shifts),
        "--output-dir", str(output_dir), "--clip-lower", str(args.clip_lower),
        "--clip-upper", str(args.clip_upper), "--cmap", args.cmap, "--dpi", str(args.dpi),
        "--point-size", str(args.point_size), "--alpha", str(args.alpha),
        "--second-peak-min-mahalanobis", str(args.second_peak_min_mahalanobis),
        "--log-transform" if args.log_transform else "--no-log-transform", "--compact"]
    if args.trial_min is not None:
        clipping_cmd += ["--trial-min", str(args.trial_min)]
    if args.trial_max is not None:
        clipping_cmd += ["--trial-max", str(args.trial_max)]
    run_checked(clipping_cmd)

    high_probability, best_score = make_separate_histograms(args, score_dir, output_dir)
    stable_map_names(output_dir, high_probability, best_score)
    run_checked([sys.executable, str(SCRIPT_DIR / "plot_activity_recall_ranking.py"),
        "--results-root", str(results_root), "--selection-dir", str(output_dir),
        "--experimental", str(args.experimental_log), "--output-dir", str(recall_output_dir),
        "--mask", str(args.mask), "--cross-index", "25",
        "--max-fraction", str(args.max_fraction), "--n-points", str(args.n_points),
        "--dpi", str(args.dpi), "--compact"])

    high_recall = recall_output_dir / (
        f"kde_highest_probability_cemc_trial_{int(high_probability['cemc']['trial']):04d}_"
        f"T_{int(high_probability['cemc']['temperature']):04d}_"
        f"random_trial_{int(high_probability['random']['trial']):04d}_activity_recall.png"
    )
    best_recall = recall_output_dir / (
        f"global_best_score_cemc_trial_{int(best_score['cemc']['trial']):04d}_"
        f"T_{int(best_score['cemc']['temperature']):04d}_"
        f"random_trial_{int(best_score['random']['trial']):04d}_activity_recall.png"
    )
    compose_all_in_one(
        output_dir / "high_probability_tau_mse_crps_3d_histogram.png",
        high_recall, output_dir / "high_probability_activity_maps.png",
        output_dir / "high_probability_all_in_one.png", args.dpi,
    )
    compose_all_in_one(
        output_dir / "best_score_tau_mse_crps_3d_histogram.png",
        best_recall, output_dir / "best_score_activity_maps.png",
        output_dir / "best_score_all_in_one.png", args.dpi,
    )

    print("\nSeparate final figures:")
    for path in (output_dir / "high_probability_tau_mse_crps_3d_histogram.png",
                 output_dir / "high_probability_activity_maps.png", recall_output_dir,
                 output_dir / "high_probability_all_in_one.png",
                 output_dir / "best_score_tau_mse_crps_3d_histogram.png",
                 output_dir / "best_score_activity_maps.png",
                 output_dir / "best_score_all_in_one.png"):
        print(path)


if __name__ == "__main__":
    main()
