#!/usr/bin/env python3
"""Random-slab OH-BE probabilities and activity volcano without BE shifts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_oh_be_distributions_no_shift as analysis


OUTPUT = analysis.BASE / "random_no_be_shift_oh_be_probability_activity_volcano"
FIGURE = OUTPUT / "random_no_be_shift_element_OH_BE_probability_activity_volcano.png"
BIN_CSV = OUTPUT / "random_no_be_shift_element_OH_BE_probability_bins.csv"
SITE_CSV = OUTPUT / "random_no_be_shift_OH_site_values.csv"
METADATA = OUTPUT / "metadata.json"
N_BINS = 100


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = analysis.parse_model(analysis.MODEL_PATH)
    surface, _ = analysis.surface_sites_and_validate(model)

    value_blocks = []
    element_blocks = []
    site_rows = []
    with tempfile.TemporaryDirectory(prefix="random_no_be_shift_") as tmp:
        tmpdir = Path(tmp)
        for run in range(20):
            occupancy = analysis.reconstruct_random(run, tmpdir)
            values, elements = analysis.evaluate(model, surface, occupancy)
            value_blocks.append(values)
            element_blocks.append(elements)
            site_activity = np.exp(
                -np.abs(values - model["e_opt"]) /
                (analysis.KB_EV_PER_K * model["activity_temperature"])
            )
            for site, (element, value, activity) in enumerate(
                    zip(elements, values, site_activity)):
                site_rows.append({
                    "run": run,
                    "slab": run + 1,
                    "OH_site": site,
                    "top_element": element,
                    "OH_BE_eV": float(value),
                    "BE_shift_eV": 0.0,
                    "site_activity": float(activity),
                })

    values = np.concatenate(value_blocks)
    elements = np.concatenate(element_blocks)
    optimum = float(model["e_opt"])
    temperature = float(model["activity_temperature"])
    lo = min(float(values.min()), optimum) - 0.08
    hi = max(float(values.max()), optimum) + 0.08
    edges = np.linspace(lo, hi, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    kernel_x = np.arange(-6, 7)
    kernel = np.exp(-0.5 * (kernel_x / 1.5) ** 2)
    kernel /= kernel.sum()
    distributions = {}
    bin_rows = []
    for element in analysis.ELEMENTS:
        selected = values[elements == element]
        counts, _ = np.histogram(selected, bins=edges)
        raw_probability = counts.astype(float) / counts.sum()
        smoothed_probability = np.convolve(raw_probability, kernel, mode="same")
        distributions[element] = smoothed_probability
        for bin_index in range(N_BINS):
            bin_rows.append({
                "element": element,
                "bin_index": bin_index,
                "bin_left_eV": float(edges[bin_index]),
                "bin_center_eV": float(centers[bin_index]),
                "bin_right_eV": float(edges[bin_index + 1]),
                "site_count": int(counts[bin_index]),
                "raw_probability": float(raw_probability[bin_index]),
                "smoothed_probability": float(smoothed_probability[bin_index]),
            })

    volcano_x = np.linspace(lo, hi, 2000)
    volcano_y = np.exp(
        -np.abs(volcano_x - optimum) /
        (analysis.KB_EV_PER_K * temperature)
    )

    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 28,
    })
    fig, ax_probability = plt.subplots(figsize=(14, 8))
    for element in analysis.ELEMENTS:
        curve = distributions[element]
        ax_probability.plot(
            centers, curve, lw=2.8,
            color=analysis.ELEMENT_COLORS[element], label=element,
        )
        ax_probability.fill_between(
            centers, curve, color=analysis.ELEMENT_COLORS[element], alpha=0.08,
        )
    ax_probability.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
    ax_probability.set_ylabel("Probability", fontsize=28)
    ax_probability.set_ylim(bottom=0.0)
    ax_probability.grid(False)

    ax_activity = ax_probability.twinx()
    ax_activity.plot(
        volcano_x, volcano_y, color="#222222", lw=3.2,
        label="Activity volcano",
    )
    ax_activity.set_ylabel("Activity", fontsize=28)
    ax_activity.set_ylim(0.0, 1.05)
    ax_probability.axvline(
        optimum, color="#D62728", ls="--", lw=2.5,
    )

    handles_left, labels_left = ax_probability.get_legend_handles_labels()
    handles_right, labels_right = ax_activity.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right, labels_left + labels_right,
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(bin_rows).to_csv(BIN_CSV, index=False)
    pd.DataFrame(site_rows).to_csv(SITE_CSV, index=False)
    element_counts = {
        element: int(np.sum(elements == element)) for element in analysis.ELEMENTS
    }
    metadata = {
        "system": "PtPdRhRuIr",
        "facet": "fcc(111)",
        "slab_method": "Random",
        "n_independent_random_slabs": 20,
        "n_OH_top_sites_per_slab": int(model["n_sites"]),
        "n_total_OH_sites": int(len(values)),
        "BE_shift_applied": False,
        "BE_shifts_eV": {element: 0.0 for element in analysis.ELEMENTS},
        "OH_BE_model": "zone1 top atom + 6 surface 1NN + 3 subsurface atoms",
        "optimal_OH_BE_eV": optimum,
        "activity_temperature_K": temperature,
        "activity_formula": "exp(-abs(OH_BE - 1.1 eV)/(k_B*298 K))",
        "x_axis_label": "Delta E_OH (eV)",
        "y_axis_label": "Probability",
        "x_axis_label_fontsize_pt": 32,
        "y_axis_label_fontsize_pt": 28,
        "axis_tick_label_fontsize_pt": 24,
        "legend_fontsize_pt": 28,
        "grid": False,
        "optimum_line_shown_in_legend": False,
        "probability_normalization": (
            "within each top element: histogram bin counts divided by the total "
            "number of Random OH sites having that top element"
        ),
        "probability_smoothing": "Gaussian kernel, sigma=1.5 bins, 13-bin support",
        "n_histogram_bins": N_BINS,
        "element_site_counts": element_counts,
        "figure": str(FIGURE),
        "bin_data": str(BIN_CSV),
        "site_data": str(SITE_CSV),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(FIGURE)


if __name__ == "__main__":
    main()
