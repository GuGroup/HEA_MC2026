#!/usr/bin/env python3
"""Create 1x2 all-temperature HBE plots with Random and CEMC only."""

import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_hbe_distributions_log_kde_peak as analysis


DISPLAY_METHODS = ("Random", "CEMC")
OUTPUT_NAME = "all_temperatures_average_element_deltaG_H_distribution_1x2_random_cemc.png"
ACTIVITY_POSITIONS = {
    "fcc100_bridge": {
        "Random": (0.96, 0.68, "right"),
        "CEMC": (0.04, 0.95, "left"),
    },
    "fcc100_hollow": {
        "Random": (0.04, 0.68, "left"),
        "CEMC": (0.04, 0.95, "left"),
    },
    "fcc110_bridge": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.96, 0.95, "right"),
    },
    "fcc111_hollow": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
    },
    "fcc111_top": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
    },
}


def plot_case(facet: str, site: str, case: str) -> Path:
    workdir = analysis.BASE / facet / site
    output_dir = (
        analysis.OUTPUT_ROOT / "by_facet_site" / case /
        analysis.ANALYSIS_1X3 / "plots_by_temperature"
    )
    output_png = output_dir / OUTPUT_NAME
    model = analysis.parse_model(workdir / "inputs/activity_model.txt")
    groups = analysis.layer_groups(workdir / "inputs/template.cif")
    _, be_shifts, _ = analysis.load_shift_data(case)
    pooled_values = {method: [] for method in analysis.METHODS}
    pooled_fractions = {method: [] for method in analysis.METHODS}

    with tempfile.TemporaryDirectory(prefix=f"replot_1x2_{case}_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            analysis.evaluate_sites(
                model, analysis.reconstruct_random(workdir, run, tmpdir), be_shifts)
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(workdir, temperature)
            evaluated = {
                "Random": random_eval,
                "CEMC": [],
                "CEMC + layer shuffle": [],
            }
            for run in range(20):
                occ = cemc[run]
                evaluated["CEMC"].append(
                    analysis.evaluate_sites(model, occ, be_shifts))
                shuffled = analysis.shuffle_occ(
                    occ,
                    groups,
                    analysis.stable_seed(
                        "FeCoNiPdPt-equi-layer-shuffle-v1",
                        f"{facet}/{site}", run, temperature,
                    ),
                )
                evaluated["CEMC + layer shuffle"].append(
                    analysis.evaluate_sites(model, shuffled, be_shifts))

            for method in analysis.METHODS:
                final_be = np.concatenate([row[2] for row in evaluated[method]])
                pooled_values[method].append(final_be - model.e_opt)
                pooled_fractions[method].append(
                    np.vstack([row[3] for row in evaluated[method]]))

    packed = {
        method: (
            np.concatenate(pooled_values[method]),
            np.vstack(pooled_fractions[method]),
        )
        for method in analysis.METHODS
    }
    activities = {
        method: analysis.activity_no_log(packed[method][0], model.activity_temperature)
        for method in DISPLAY_METHODS
    }
    # Preserve the exact common x-grid used by the original 1x3 publication plot.
    combined = np.concatenate([packed[method][0] for method in analysis.METHODS])
    lo = min(float(combined.min()), analysis.OPTIMAL_DELTA_G_H_EV) - 0.06
    hi = max(float(combined.max()), analysis.OPTIMAL_DELTA_G_H_EV) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.linewidth": 1.2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4), sharex=True, sharey=True)
    for method, ax in zip(DISPLAY_METHODS, axes):
        values, fractions = packed[method]
        for ei, element in enumerate(analysis.ELEMENTS):
            density = analysis.smoothed_weighted_density(
                values, fractions[:, ei], edges, len(values))
            ax.plot(
                centers, density, lw=2.7,
                color=analysis.ELEMENT_COLORS[element], label=element,
            )
        ax.axvline(
            analysis.OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.1, color="black",
            label=r"Optimal $\Delta G_{\mathrm{H}}$",
        )
        ax.set_title("Homogeneous" if method == "Random" else method, pad=10)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.tick_params(width=1.2, length=5)
        ax.grid(False)
        activity_x, activity_y, horizontal_alignment = ACTIVITY_POSITIONS[case][method]
        ax.text(
            activity_x, activity_y, f"Activity = {activities[method]:.6f}",
            transform=ax.transAxes, ha=horizontal_alignment, va="top", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3.0),
            zorder=90,
        )

    axes[0].set_ylabel("Probability density")
    handles, labels = axes[0].get_legend_handles_labels()
    kept = [(handle, label) for handle, label in zip(handles, labels)
            if not label.startswith("Optimal")]
    handles, labels = zip(*kept)
    legend = axes[0].legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98),
        ncol=2, frameon=True, framealpha=0.92, facecolor="white", edgecolor="none",
        columnspacing=1.4, handlelength=2.4, fontsize=18,
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[case for _, _, case in analysis.COMBOS])
    args = parser.parse_args()
    for facet, site, case in analysis.COMBOS:
        if args.case is not None and case != args.case:
            continue
        output_png = plot_case(facet, site, case)
        print(output_png, flush=True)


if __name__ == "__main__":
    main()
