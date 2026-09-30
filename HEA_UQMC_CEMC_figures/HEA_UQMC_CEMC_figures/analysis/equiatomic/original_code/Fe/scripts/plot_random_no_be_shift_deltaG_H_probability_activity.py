#!/usr/bin/env python3
"""Element-resolved random-slab deltaG_H distributions by facet/site."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
OUTPUT_ROOT = BASE / "random_no_be_shift_deltaG_H_probability_activity_volcano" / "by_facet_site"
ANALYSIS_SOURCE = BASE / "scripts/analyze_hbe_distributions.py"
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
ELEMENT_COLORS = {
    "Fe": "#E31A1C", "Co": "#FF9900", "Ni": "#2CA02C",
    "Pd": "#FFD700", "Pt": "#1F77B4",
}
CASES = (
    ("100", "hollow", "fcc100_hollow"),
    ("100", "bridge", "fcc100_bridge"),
    ("110", "bridge", "fcc110_bridge"),
    ("111", "hollow", "fcc111_hollow"),
    ("111", "top", "fcc111_top"),
)
N_BINS = 90


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("hbe_analysis", ANALYSIS_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def process_case(analysis, facet: str, site: str, case: str, tmpdir: Path) -> Path:
    output = OUTPUT_ROOT / case
    output.mkdir(parents=True, exist_ok=True)
    workdir = BASE / facet / site
    model = analysis.parse_model(workdir / "inputs/activity_model.txt")

    values_by_run = []
    fractions_by_run = []
    site_rows = []
    for run in range(20):
        occupancy = analysis.reconstruct_random(workdir, run, tmpdir)
        linear_be, shifts, final_be, fractions, _ = analysis.evaluate_sites(
            model, occupancy, np.zeros(len(ELEMENTS), dtype=float)
        )
        if not np.allclose(shifts, 0.0) or not np.allclose(final_be, linear_be):
            raise RuntimeError(f"Unexpected nonzero BE shift for {case}")
        delta_g_h = final_be - model.e_opt
        values_by_run.append(delta_g_h)
        fractions_by_run.append(fractions)
        site_activity = np.exp(
            -np.abs(delta_g_h) /
            (analysis.KB_EV_PER_K * model.activity_temperature)
        )
        for site_index in range(model.n_sites):
            row = {
                "facet": f"fcc({facet})", "site": site, "run": run,
                "slab": run + 1, "site_index": site_index,
                "deltaE_H_eV": float(final_be[site_index]),
                "reference_optimal_deltaE_H_eV": float(model.e_opt),
                "deltaG_H_eV": float(delta_g_h[site_index]),
                "BE_shift_eV": 0.0,
                "site_activity": float(site_activity[site_index]),
            }
            row.update({
                f"zone1_fraction_{element}": float(fractions[site_index, ei])
                for ei, element in enumerate(ELEMENTS)
            })
            site_rows.append(row)

    values = np.concatenate(values_by_run)
    fractions = np.vstack(fractions_by_run)
    lo = min(float(values.min()), 0.0) - 0.06
    hi = max(float(values.max()), 0.0) + 0.06
    edges = np.linspace(lo, hi, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])

    curves = {}
    bin_rows = []
    element_stats = {}
    for element_index, element in enumerate(ELEMENTS):
        weights = fractions[:, element_index]
        weighted_counts, _ = np.histogram(values, bins=edges, weights=weights)
        raw_density = weighted_counts / (len(values) * width)
        smoothed_density = analysis.smoothed_weighted_density(
            values, weights, edges, len(values)
        )
        curves[element] = smoothed_density
        positive = weights > 0
        weight_sum = float(weights.sum())
        weighted_mean = float(np.average(values[positive], weights=weights[positive]))
        weighted_sd = float(np.sqrt(np.average(
            (values[positive] - weighted_mean) ** 2,
            weights=weights[positive],
        )))
        element_stats[element] = {
            "integrated_fractional_contribution": weight_sum / len(values),
            "weighted_mean_deltaG_H_eV": weighted_mean,
            "weighted_sd_deltaG_H_eV": weighted_sd,
        }
        for bin_index in range(N_BINS):
            bin_rows.append({
                "facet": f"fcc({facet})", "site": site, "element": element,
                "bin_index": bin_index,
                "bin_left_eV": float(edges[bin_index]),
                "bin_center_eV": float(centers[bin_index]),
                "bin_right_eV": float(edges[bin_index + 1]),
                "weighted_site_count": float(weighted_counts[bin_index]),
                "raw_probability_density_eV_inv": float(raw_density[bin_index]),
                "smoothed_probability_density_eV_inv": float(smoothed_density[bin_index]),
            })

    activity = analysis.activity_no_log(values, model.activity_temperature)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 24,
        "axes.labelsize": 32,
        "xtick.labelsize": 24, "ytick.labelsize": 24,
        "legend.fontsize": 28, "axes.linewidth": 1.2,
    })
    fig, ax = plt.subplots(figsize=(14, 8))
    for element in ELEMENTS:
        ax.plot(centers, curves[element], lw=2.8,
                color=ELEMENT_COLORS[element], label=element)
        ax.fill_between(centers, curves[element],
                        color=ELEMENT_COLORS[element], alpha=0.08)
    ax.axvline(0.0, ls="--", lw=2.5, color="#D62728")
    ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
    ax.set_ylabel("Probability", fontsize=28)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(width=1.2, length=5)
    ax.grid(False)

    volcano_x = np.linspace(lo, hi, 2000)
    volcano_y = np.exp(
        -np.abs(volcano_x) /
        (analysis.KB_EV_PER_K * model.activity_temperature)
    )
    ax_activity = ax.twinx()
    ax_activity.plot(volcano_x, volcano_y, color="#222222", lw=3.2,
                     label="Activity volcano")
    ax_activity.set_ylabel("Activity", fontsize=28)
    ax_activity.set_ylim(0.0, 1.05)
    ax_activity.tick_params(width=1.2, length=5)
    ax_activity.grid(False)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_activity.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right, labels_left + labels_right,
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    figure = output / "random_element_deltaG_H_distribution.png"
    fig.savefig(figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    site_csv = output / "random_deltaG_H_site_values.csv"
    bin_csv = output / "element_deltaG_H_probability_density_bins.csv"
    pd.DataFrame(site_rows).to_csv(site_csv, index=False)
    pd.DataFrame(bin_rows).to_csv(bin_csv, index=False)
    metadata = {
        "system": "FeCoNiPdPt", "facet": f"fcc({facet})", "site": site,
        "slab_method": "Random (Homogeneous)",
        "n_independent_random_slabs": 20,
        "n_adsorption_sites_per_slab": int(model.n_sites),
        "n_total_adsorption_sites": int(len(values)),
        "BE_shift_applied": False, "composition_shift_applied": False,
        "reference_optimal_deltaE_H_eV": float(model.e_opt),
        "deltaG_H_formula": f"deltaE_H - ({model.e_opt:+.3f} eV)",
        "optimal_deltaG_H_eV": 0.0,
        "activity_temperature_K": float(model.activity_temperature),
        "activity_formula": "mean(exp(-abs(deltaG_H)/(k_B*T_activity)))",
        "activity_no_log": float(activity),
        "activity_volcano_plotted": True,
        "optimum_line_shown_in_legend": False,
        "y_axis_label": "Probability",
        "x_axis_label_fontsize_pt": 32,
        "y_axis_label_fontsize_pt": 28,
        "axis_tick_label_fontsize_pt": 24,
        "legend_fontsize_pt": 28,
        "grid": False,
        "element_curve_weighting": (
            "zone1 fractional contribution; top is one-hot, bridge/hollow sites "
            "are fractionally assigned to their zone1 atoms"
        ),
        "distribution_normalization": (
            "weighted count / (all adsorption sites * bin width); each curve area "
            "equals the element's zone1 fractional contribution"
        ),
        "probability_smoothing": "Gaussian kernel, sigma=1.5 bins, 13-bin support",
        "n_histogram_bins": N_BINS,
        "element_statistics": element_stats,
        "figure": str(figure), "bin_data": str(bin_csv), "site_data": str(site_csv),
        "source_analysis_code": str(ANALYSIS_SOURCE),
        "reference_plot_code": str(
            BASE / "new_BE_shift_completed_sites_log_KDE_peak/scripts/"
            "plot_most_probable_trial_shift_1x3_all_temperatures.py"
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Random (Homogeneous) FeCoNiPdPt equi-atomic slab distribution.\n"
        f"Facet/site: fcc({facet}) {site}\n"
        f"DeltaG_H = DeltaE_H - ({model.e_opt:+.3f} eV)\n"
        "No BE shift or composition shift was applied.\n"
        "Element curves use zone1 fractional contributions.\n",
        encoding="utf-8",
    )
    print(figure, flush=True)
    return figure


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_module()
    with tempfile.TemporaryDirectory(prefix="random_element_dgh_") as tmp:
        tmpdir = Path(tmp)
        for facet, site, case in CASES:
            process_case(analysis, facet, site, case, tmpdir)


if __name__ == "__main__":
    main()
