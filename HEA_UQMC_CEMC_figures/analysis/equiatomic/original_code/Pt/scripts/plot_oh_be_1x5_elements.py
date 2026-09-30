#!/usr/bin/env python3
"""Additional 1x5 element-panel OH-BE plots; preserves existing 1x3 plots."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_oh_be_distributions as analysis


OUTPUT = analysis.BASE / "oh_be_distribution_analysis_be_shift_1x5_elements"
PLOTS = OUTPUT / "plots_by_temperature"
COLORS = {"Random": "#555555", "CEMC": "#2468B4", "CEMC + layer shuffle": "#E68613"}


def plot(packed, activities, title, destination, optimum):
    combined = np.concatenate([packed[m][0] for m in analysis.METHODS])
    lo = min(float(combined.min()), optimum) - 0.08
    hi = max(float(combined.max()), optimum) + 0.08
    edges = np.linspace(lo, hi, 101)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    })
    fig, axes = plt.subplots(1, 5, figsize=(25, 6.2), sharex=True, sharey=True)
    for element, ax in zip(analysis.ELEMENTS, axes):
        for method in analysis.METHODS:
            values, labels = packed[method]
            density, _ = analysis.smooth_density(values, labels, element, edges)
            ax.plot(centers, density, lw=2.6, color=COLORS[method], label=method)
            ax.fill_between(centers, density, color=COLORS[method], alpha=0.10)
        ax.axvline(optimum, color="black", ls="--", lw=2.2,
                   label="Optimal OH BE (1.1 eV)" if element == analysis.ELEMENTS[0] else None)
        ax.set_title(element, pad=9)
        ax.set_xlabel("OH BE (eV)")
        ax.grid(alpha=0.20)
        ax.tick_params(width=1.2, length=5)
    axes[0].set_ylabel("Surface-fraction-weighted site density (eV$^{-1}$)")
    activity_text = " | ".join(
        f"{method}: {np.mean(activities[method]):.6f} ± {np.std(activities[method]):.6f}"
        for method in analysis.METHODS)
    fig.suptitle(f"{title}\nActivity (slab mean ± SD; no natural log) | {activity_text}",
                 fontsize=18, y=1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    model = analysis.parse_model(analysis.MODEL_PATH)
    surface, layers = analysis.surface_sites_and_validate(model)
    shifts = analysis.SELECTED_BE_SHIFTS_EV
    packed_by_temperature = {}
    activity_by_temperature = {}
    with tempfile.TemporaryDirectory(prefix="ptpdirrhru_1x5_") as tmp:
        tmpdir = Path(tmp)
        random_evaluated = [
            analysis.evaluate(model, surface, analysis.reconstruct_random(run, tmpdir), shifts)
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc(temperature)
            evaluated = {
                "Random": random_evaluated,
                "CEMC": [analysis.evaluate(model, surface, cemc[run], shifts) for run in range(20)],
                "CEMC + layer shuffle": [
                    analysis.evaluate(model, surface, analysis.shuffle_within_layers(
                        cemc[run], layers, analysis.stable_seed(
                            "PtPdRhRuIr-equi-atomic-layer-shuffle-v1", run, temperature)), shifts)
                    for run in range(20)
                ],
            }
            packed = {
                method: (np.concatenate([x[0] for x in evaluated[method]]),
                         np.concatenate([x[1] for x in evaluated[method]]))
                for method in analysis.METHODS
            }
            activities = {
                method: np.asarray([
                    analysis.activity_no_log(x[0], model["e_opt"], model["activity_temperature"])
                    for x in evaluated[method]])
                for method in analysis.METHODS
            }
            packed_by_temperature[temperature] = packed
            activity_by_temperature[temperature] = activities
            label = "final 298 K" if temperature == 298 else f"{temperature} K"
            stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
            plot(packed, activities,
                 f"PtPdIrRhRu equi-atomic fcc(111): {label}\nOH binding-energy distribution",
                 PLOTS / f"{stem}_element_panels_OH_BE_distribution.png", model["e_opt"])
    packed_all = {
        method: (np.concatenate([packed_by_temperature[t][method][0] for t in analysis.TEMPERATURES]),
                 np.concatenate([packed_by_temperature[t][method][1] for t in analysis.TEMPERATURES]))
        for method in analysis.METHODS
    }
    activities_all = {
        method: np.asarray([
            np.mean([activity_by_temperature[t][method][run] for t in analysis.TEMPERATURES])
            for run in range(20)])
        for method in analysis.METHODS
    }
    plot(packed_all, activities_all,
         "PtPdIrRhRu equi-atomic fcc(111): average over all 19 temperatures\n"
         "OH binding-energy distribution",
         PLOTS / "all_temperatures_average_element_panels_OH_BE_distribution.png",
         model["e_opt"])
    metadata = {
        "system": "PtPdIrRhRu", "layout": "1 row x 5 element panels",
        "series_per_panel": list(analysis.METHODS), "BE_shift_applied": True,
        "selected_BE_shifts_eV": shifts, "optimal_OH_BE_eV": model["e_opt"],
        "n_png": 20, "n_pdf": 0,
        "existing_1x3_results_preserved": True,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
