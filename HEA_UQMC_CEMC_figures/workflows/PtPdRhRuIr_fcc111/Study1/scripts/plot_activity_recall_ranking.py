#!/usr/bin/env python3
"""Plot log-activity top-fraction recall for KDE and global-best representatives."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPACT = False

@dataclass(frozen=True)
class RecallCase:
    name: str
    title: str
    mc_trial: int
    mc_temperature: int
    mc_shard: str
    random_trial: int
    random_shard: str
    clip_lower: float
    clip_upper: float


def read_zarr_array(store: Path, name: str) -> np.ndarray:
    array_dir = store / name
    with (array_dir / ".zarray").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    if chunks != shape:
        raise ValueError(f"Expected one chunk in {array_dir}; got {chunks} vs {shape}")
    chunk_path = array_dir / ".".join("0" for _ in shape)
    values = np.fromfile(chunk_path, dtype=np.dtype(metadata["dtype"]))
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Unexpected chunk size in {chunk_path}")
    return values.reshape(shape).astype(np.float64)


def load_mc_log_activity(results_root: Path, shard: str, trial: int, temperature: int) -> np.ndarray:
    store = results_root / shard / "trial_zarr" / f"trial_{trial:06d}" / "mc.zarr"
    with (store / ".zattrs").open(encoding="utf-8") as handle:
        temperatures = [int(round(float(v))) for v in json.load(handle)["temperatures"]]
    if temperature not in temperatures:
        raise ValueError(f"Temperature {temperature} K is absent from {store}")
    return read_zarr_array(store, "activity_mean")[temperatures.index(temperature)]


def load_random_log_activity(results_root: Path, shard: str, trial: int) -> np.ndarray:
    store = results_root / shard / "trial_zarr" / f"trial_{trial:06d}" / "random.zarr"
    return read_zarr_array(store, "activity_mean")


def load_experimental(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    keys = sorted(int(key) for key in records)
    if keys != list(range(len(keys))):
        raise ValueError(f"Composition indices in {path} must be contiguous from zero")
    values = np.asarray([records[str(i)] for i in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite experimental activity in {path}")
    return values


def center_cross_keep(mask: np.ndarray, cross_index: int) -> np.ndarray:
    cross = np.zeros(mask.shape, dtype=bool)
    cross[cross_index, :] = True
    cross[:, cross_index] = True
    return (~cross)[~mask]


def clip_and_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predicted activity contains non-finite values")
    low_bound, high_bound = np.percentile(values, [lower, upper])
    clipped = np.clip(values, low_bound, high_bound)
    span = float(np.max(clipped) - np.min(clipped))
    return np.zeros_like(clipped) if span < 1e-30 else -(clipped - np.min(clipped)) / span


def recall_curve(experimental: np.ndarray, prediction: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Recall among the most active fraction; -1 is most active after scaling."""
    if experimental.shape != prediction.shape:
        raise ValueError(f"Activity shape mismatch: {experimental.shape} vs {prediction.shape}")
    exp_order = np.argsort(experimental, kind="stable")
    pred_order = np.argsort(prediction, kind="stable")
    n = len(experimental)
    recall = np.zeros_like(fractions)
    for i, fraction in enumerate(fractions):
        if fraction <= 0:
            continue
        count = min(n, max(1, int(np.ceil(fraction * n))))
        exp_top = np.zeros(n, dtype=bool)
        exp_top[exp_order[:count]] = True
        recall[i] = float(exp_top[pred_order[:count]].sum()) / count
    return recall


def load_cases(selection_dir: Path) -> list[RecallCase]:
    kde = pd.read_csv(selection_dir / "modal_tau_mse_crps_representative_trials_kde.csv").iloc[0]
    best = pd.read_csv(selection_dir / "global_best_score_cemc_random.csv")
    best_mc = best[best["method"] == "cemc"].iloc[0]
    best_random = best[best["method"] == "random"].iloc[0]
    return [
        RecallCase(
            "kde_highest_probability", "Highest-probability KDE representatives",
            int(kde["mc_trial"]), int(kde["mc_temperature"]), str(kde["mc_source_shard"]),
            int(kde["random_trial"]), str(kde["random_source_shard"]),
            float(kde["clip_lower_percentile"]), float(kde["clip_upper_percentile"]),
        ),
        RecallCase(
            "global_best_score", "Global best-score representatives",
            int(best_mc["trial"]), int(best_mc["temperature"]), str(best_mc["source_shard"]),
            int(best_random["trial"]), str(best_random["source_shard"]),
            float(best_mc["clip_lower_percentile"]), float(best_mc["clip_upper_percentile"]),
        ),
    ]


def plot_recall(fractions: np.ndarray, cemc: np.ndarray, random: np.ndarray,
                case: RecallCase, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(fractions * 100, cemc, color="#4C78A8", lw=2.4, label="CEMC")
    ax.plot(fractions * 100, random, color="#F58518", lw=2.4, label="Homogeneous")
    ax.plot(fractions * 100, fractions, "k--", lw=2.0, label="Random baseline")
    ax.set(xlim=(0, fractions[-1] * 100), ylim=(0, 1),
           xlabel="Top fraction tested (%)", ylabel="Recall of experimental active region")
    if not COMPACT:
        ax.set_title(f"{case.title}\nCEMC trial {case.mc_trial}, {case.mc_temperature} K; "
                     f"Homogeneous trial {case.random_trial}")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(axis="both", direction="in", length=6, width=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global COMPACT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--selection-dir", type=Path,
                        default=Path("results/modal_tau_mse_crps_activity_map_kde_clipped_scaled_log"))
    parser.add_argument("--experimental", type=Path, default=Path("orr_log_matched_activity.json"))
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--cross-index", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("results/activity_recall_ranking_log"))
    parser.add_argument("--max-fraction", type=float, default=0.60)
    parser.add_argument("--n-points", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--compact", action="store_true", help="Use larger, minimal text for composite figures")
    args = parser.parse_args()
    COMPACT = args.compact
    if COMPACT:
        plt.rcParams.update({"font.size": 16, "axes.labelsize": 17,
                             "xtick.labelsize": 14, "ytick.labelsize": 14,
                             "legend.fontsize": 15})
    if not (0 < args.max_fraction <= 1):
        parser.error("--max-fraction must be in (0, 1]")
    if args.n_points < 2:
        parser.error("--n-points must be at least 2")

    experimental = load_experimental(args.experimental)
    evaluation_keep = center_cross_keep(np.load(args.mask).astype(bool), args.cross_index)
    if len(evaluation_keep) != len(experimental):
        raise ValueError("Static mask and experimental activity have different packed lengths")
    experimental = experimental[evaluation_keep]
    fractions = np.linspace(0, args.max_fraction, args.n_points + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in load_cases(args.selection_dir):
        cemc = clip_and_scale(
            load_mc_log_activity(args.results_root, case.mc_shard, case.mc_trial, case.mc_temperature)[evaluation_keep],
            case.clip_lower, case.clip_upper,
        )
        random = clip_and_scale(
            load_random_log_activity(args.results_root, case.random_shard, case.random_trial)[evaluation_keep],
            case.clip_lower, case.clip_upper,
        )
        cemc_recall = recall_curve(experimental, cemc, fractions)
        random_recall = recall_curve(experimental, random, fractions)
        stem = (f"{case.name}_cemc_trial_{case.mc_trial:04d}_T_{case.mc_temperature:04d}_"
                f"random_trial_{case.random_trial:04d}_activity_recall")
        plot_recall(fractions, cemc_recall, random_recall, case,
                    args.output_dir / f"{stem}.png", args.dpi)
        pd.DataFrame({
            "case": case.name, "cemc_trial": case.mc_trial,
            "cemc_temperature": case.mc_temperature, "random_trial": case.random_trial,
            "top_fraction": fractions, "baseline_recall": fractions,
            "cemc_recall": cemc_recall, "random_recall": random_recall,
        }).to_csv(args.output_dir / f"{stem}.csv", index=False)
        print(f"{case.name}: CEMC trial={case.mc_trial}, T={case.mc_temperature} K; "
              f"Homogeneous trial={case.random_trial}; recall@10% "
              f"CEMC={np.interp(.10, fractions, cemc_recall):.4f}, "
              f"Homogeneous={np.interp(.10, fractions, random_recall):.4f}")
        print(f"wrote {args.output_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
