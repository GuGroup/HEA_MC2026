#!/usr/bin/env python3
"""Additional 1x5 element-panel deltaG_H plots; preserves existing 1x3 plots."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_hbe_distributions_new_be_shift as analysis


COLORS = {"Random": "#555555", "CEMC": "#2468B4", "CEMC + layer shuffle": "#E68613"}


def plot(packed, activities, title, destination, optimum_hbe):
    combined = np.concatenate([packed[m][0] for m in analysis.METHODS])
    lo = min(float(combined.min()), 0.0) - 0.06
    hi = max(float(combined.max()), 0.0) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    })
    fig, axes = plt.subplots(1, 5, figsize=(25, 6.2), sharex=True, sharey=True)
    densities = {method: [] for method in analysis.METHODS}
    for method in analysis.METHODS:
        values, fractions = packed[method]
        for ei in range(len(analysis.ELEMENTS)):
            densities[method].append(analysis.smoothed_weighted_density(
                values, fractions[:, ei], edges, len(values)))
    for ei, (element, ax) in enumerate(zip(analysis.ELEMENTS, axes)):
        for method in analysis.METHODS:
            peak = float(np.max(densities[method][ei]))
            density = densities[method][ei] / peak if peak > 0.0 else densities[method][ei]
            ax.plot(centers, density, lw=2.6, color=COLORS[method], label=method)
            ax.fill_between(centers, density, color=COLORS[method], alpha=0.10)
        ax.axvline(0.0, color="black", ls="--", lw=2.2,
                   label=r"Optimal $\Delta G_{\mathrm{H}}$" if ei == 0 else None)
        ax.set_title(element, pad=9)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.grid(alpha=0.20)
        ax.tick_params(width=1.2, length=5)
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Normalized density (each curve peak = 1)")
    activity_text = " | ".join(
        f"{method}: {np.mean(activities[method]):.6f} ± {np.std(activities[method]):.6f}"
        for method in analysis.METHODS)
    fig.suptitle(f"{title}\n$\\Delta G_{{\\mathrm{{H}}}}$ = H BE - ({optimum_hbe:+.3f} eV)"
                 f"\nActivity (slab mean ± SD; no natural log) | {activity_text}",
                 fontsize=18, y=1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_case(facet, site, case):
    combo = f"{facet}/{site}"
    workdir = analysis.BASE / facet / site
    output = workdir / "hbe_distribution_analysis_log_activity_top100_new_BE_shift_1x5_elements"
    plots = output / "plots_by_temperature"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    model = analysis.parse_model(workdir / "inputs/activity_model.txt")
    groups = analysis.layer_groups(workdir / "inputs/template.cif")
    _, shifts, _ = analysis.load_shift_data(case)
    packed_by_temperature = {}
    activity_by_temperature = {}
    with tempfile.TemporaryDirectory(prefix=f"fecoonipdpt_1x5_{facet}_{site}_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            analysis.evaluate_sites(model, analysis.reconstruct_random(workdir, run, tmpdir), shifts)
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(workdir, temperature)
            evaluated = {"Random": random_eval, "CEMC": [], "CEMC + layer shuffle": []}
            for run in range(20):
                occ = cemc[run]
                evaluated["CEMC"].append(analysis.evaluate_sites(model, occ, shifts))
                shuffled = analysis.shuffle_occ(
                    occ, groups,
                    analysis.stable_seed("FeCoNiPdPt-equi-layer-shuffle-v1", combo, run, temperature))
                evaluated["CEMC + layer shuffle"].append(
                    analysis.evaluate_sites(model, shuffled, shifts))
            packed = {}
            activities = {}
            for method in analysis.METHODS:
                delta_by_run = [x[2] - model.e_opt for x in evaluated[method]]
                packed[method] = (
                    np.concatenate(delta_by_run),
                    np.vstack([x[3] for x in evaluated[method]]),
                )
                activities[method] = np.asarray([
                    analysis.activity_no_log(values, model.activity_temperature)
                    for values in delta_by_run])
            packed_by_temperature[temperature] = packed
            activity_by_temperature[temperature] = activities
            label = "final 298 K" if temperature == 298 else f"{temperature} K"
            stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
            plot(packed, activities, f"FeCoNiPdPt fcc({facet}) {site}: {label}",
                 plots / f"{stem}_element_panels_deltaG_H_distribution.png", model.e_opt)
    packed_all = {
        method: (np.concatenate([packed_by_temperature[t][method][0] for t in analysis.TEMPERATURES]),
                 np.vstack([packed_by_temperature[t][method][1] for t in analysis.TEMPERATURES]))
        for method in analysis.METHODS
    }
    activities_all = {
        method: np.asarray([
            np.mean([activity_by_temperature[t][method][run] for t in analysis.TEMPERATURES])
            for run in range(20)])
        for method in analysis.METHODS
    }
    plot(packed_all, activities_all,
         f"FeCoNiPdPt fcc({facet}) {site}: average over all 19 temperatures",
         plots / "all_temperatures_average_element_panels_deltaG_H_distribution.png",
         model.e_opt)
    metadata = {
        "system": "FeCoNiPdPt", "facet": facet, "site": site,
        "layout": "1 row x 5 element panels", "series_per_panel": list(analysis.METHODS),
        "BE_shift_applied": True, "selected_BE_shifts_eV": dict(zip(analysis.ELEMENTS, shifts)),
        "BE_shift_source_workbook": str(analysis.SHIFT_WORKBOOK),
        "BE_shift_source_sheet": "Fe_BE", "BE_shift_source_column": "Exact BE shift (eV)",
        "density_normalization": (
            "Within each element panel, divide each slab-method density curve by its own "
            "maximum; Random, CEMC, and layer-shuffled curves each peak at 1."
        ),
        "optimal_deltaG_H_eV": 0.0, "n_png": 20, "n_pdf": 0,
        "existing_1x3_results_preserved": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    metadata = [analyze_case(facet, site, case) for facet, site, case in analysis.COMBOS]
    overall = {
        "n_cases": len(metadata), "n_png": 20 * len(metadata), "n_pdf": 0,
        "layout": "1 row x 5 element panels",
        "density_normalization": "each method curve within each element panel peaks at 1",
        "cases": metadata,
    }
    (analysis.BASE / "H_BE_distribution_log_activity_top100_new_BE_shift_1x5_analysis.json").write_text(
        json.dumps(overall, indent=2) + "\n")
    print(json.dumps({k: v for k, v in overall.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
