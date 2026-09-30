#!/usr/bin/env python3
"""Plot one PtPdIrRhRu 1,400-composition continuous-recall figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_activity_recall_ranking as recall


ROOT = Path("/home/ktg0829/project/HEA/CEMC/UQMC_new_composition_shift")
RESULTS_ROOT = ROOT / "results"
SELECTION_DIR = (
    RESULTS_ROOT / "modal_tau_mse_crps_activity_map_kde_clipped_scaled_log"
)
EXPERIMENTAL = ROOT / "orr_log_matched_activity.json"
MASK = Path(
    "/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy"
)
OUTPUT = (
    SELECTION_DIR
    / "PtPdIrRhRu_1400_log_activity_continuous_recall_analysis.png"
)
SYSTEM_COMPOSITIONS = 1400


def main() -> None:
    # Use the modal/highest-probability KDE representative pair selected by the
    # existing 1,400-composition recall-analysis workflow.
    case = recall.load_cases(SELECTION_DIR)[0]
    evaluation_keep = recall.center_cross_keep(np.load(MASK).astype(bool), 25)
    experimental = recall.load_experimental(EXPERIMENTAL)
    if len(evaluation_keep) != len(experimental):
        raise ValueError("Static mask and experimental activity lengths differ")
    experimental = experimental[evaluation_keep]
    n_evaluated = len(experimental)

    cemc = recall.clip_and_scale(
        recall.load_mc_log_activity(
            RESULTS_ROOT, case.mc_shard, case.mc_trial, case.mc_temperature
        )[evaluation_keep],
        case.clip_lower,
        case.clip_upper,
    )
    random = recall.clip_and_scale(
        recall.load_random_log_activity(
            RESULTS_ROOT, case.random_shard, case.random_trial
        )[evaluation_keep],
        case.clip_lower,
        case.clip_upper,
    )

    # Match the FeCoNiPdPt continuous analysis: evaluate every integer top_n
    # through 60% instead of using only a coarse set of sampled fractions.
    top_n = np.arange(1, int(np.floor(0.60 * n_evaluated)) + 1)
    fractions = top_n / n_evaluated
    cemc_recall = recall.recall_curve(experimental, cemc, fractions)
    random_recall = recall.recall_curve(experimental, random, fractions)

    with plt.rc_context({
        "font.size": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    }):
        fig, ax = plt.subplots(figsize=(11.2, 8.4))
        x = 100.0 * fractions
        ax.plot(x, cemc_recall, color="#0072B2", linewidth=2.8, label="CEMC")
        ax.plot(
            x, random_recall, color="#D55E00", linewidth=2.8,
            label="Homogeneous",
        )
        ax.plot(
            x, fractions, color="black", linewidth=2.2,
            linestyle="--", label="Baseline",
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Top fraction tested (%)", fontsize=28)
        ax.set_ylabel("Recall of experimental active region", fontsize=28)
        ax.tick_params(axis="both", labelsize=24)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=24)
        fig.tight_layout()
        fig.savefig(OUTPUT, dpi=250, bbox_inches="tight")
        plt.close(fig)

    print(OUTPUT)
    print(
        f"system_n={SYSTEM_COMPOSITIONS}, evaluated_n={n_evaluated}, "
        f"CEMC trial={case.mc_trial}, "
        f"T={case.mc_temperature} K, Random trial={case.random_trial}"
    )


if __name__ == "__main__":
    main()
