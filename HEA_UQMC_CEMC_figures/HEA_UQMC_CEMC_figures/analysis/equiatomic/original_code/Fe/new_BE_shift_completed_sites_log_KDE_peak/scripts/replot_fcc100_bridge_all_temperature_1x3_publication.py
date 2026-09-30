#!/usr/bin/env python3
"""Replot the fcc(100)-bridge all-temperature 1x3 HBE distribution."""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_hbe_distributions_log_kde_peak as analysis


FACET = "100"
SITE = "bridge"
CASE = "fcc100_bridge"
WORKDIR = analysis.BASE / FACET / SITE
OUTPUT_DIR = (
    analysis.OUTPUT_ROOT / "by_facet_site" / CASE /
    analysis.ANALYSIS_1X3 / "plots_by_temperature"
)
OUTPUT_PNG = OUTPUT_DIR / "all_temperatures_average_element_deltaG_H_distribution.png"


def main() -> None:
    model = analysis.parse_model(WORKDIR / "inputs/activity_model.txt")
    groups = analysis.layer_groups(WORKDIR / "inputs/template.cif")
    _, be_shifts, _ = analysis.load_shift_data(CASE)
    pooled_values = {method: [] for method in analysis.METHODS}
    pooled_fractions = {method: [] for method in analysis.METHODS}

    with tempfile.TemporaryDirectory(prefix="replot_fcc100_bridge_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            analysis.evaluate_sites(
                model, analysis.reconstruct_random(WORKDIR, run, tmpdir), be_shifts)
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(WORKDIR, temperature)
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
                    occ, groups,
                    analysis.stable_seed(
                        "FeCoNiPdPt-equi-layer-shuffle-v1",
                        f"{FACET}/{SITE}", run, temperature,
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
    combined = np.concatenate([packed[method][0] for method in analysis.METHODS])
    lo = min(float(combined.min()), analysis.OPTIMAL_DELTA_G_H_EV) - 0.06
    hi = max(float(combined.max()), analysis.OPTIMAL_DELTA_G_H_EV) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4), sharex=True, sharey=True)
    for method, ax in zip(analysis.METHODS, axes):
        values, fractions = packed[method]
        for ei, element in enumerate(analysis.ELEMENTS):
            density = analysis.smoothed_weighted_density(
                values, fractions[:, ei], edges, len(values))
            ax.plot(
                centers, density, lw=2.6,
                color=analysis.ELEMENT_COLORS[element], label=element,
            )
        ax.axvline(
            analysis.OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.0, color="black",
            label=r"Optimal $\Delta G_{\mathrm{H}}$",
        )
        ax.set_title(method, pad=9)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.tick_params(width=1.2, length=5)
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("Probability density (eV$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=6, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
