#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_hbe_distributions as analysis


ROOT = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/100/bridge/hbe_distribution_analysis")
OUTPUT = ROOT / "plots_by_temperature/all_temperatures_average_element_deltaG_H_distribution.png"


def main() -> None:
    metadata = json.loads((ROOT / "analysis_metadata.json").read_text())
    reference_hbe = float(metadata["reference_optimal_H_BE_eV"])
    activity_temperature = float(metadata["activity_temperature_K"])
    shift_table = pd.read_csv(ROOT / "selected_shifts_and_final_composition.csv").set_index("element")
    be_shifts = shift_table.loc[list(analysis.ELEMENTS), "selected_be_shift_exact_eV"].to_numpy(float)

    usecols = (["method", "final_H_BE_eV"] +
               [f"zone1_fraction_{element}" for element in analysis.ELEMENTS])
    frames = [pd.read_csv(path, usecols=usecols)
              for path in sorted((ROOT / "site_data_by_temperature").glob("*.csv.gz"))]
    data = pd.concat(frames, ignore_index=True)

    packed = {}
    activities = {}
    for method in analysis.METHODS:
        subset = data.loc[data["method"] == method]
        delta_g_h = subset["final_H_BE_eV"].to_numpy(float) - reference_hbe
        fractions = subset[[f"zone1_fraction_{e}" for e in analysis.ELEMENTS]].to_numpy(float)
        packed[method] = (delta_g_h, fractions)
        activities[method] = analysis.activity_no_log(delta_g_h, activity_temperature)

    combined = np.concatenate([packed[method][0] for method in analysis.METHODS])
    lo = min(float(combined.min()), 0.0) - 0.06
    hi = max(float(combined.max()), 0.0) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
    fig, axes = plt.subplots(1, 5, figsize=(25, 6.2), sharex=True, sharey=True)
    for ei, (element, ax) in enumerate(zip(analysis.ELEMENTS, axes)):
        for method in analysis.METHODS:
            values, fractions = packed[method]
            density = analysis.smoothed_weighted_density(values, fractions[:, ei], edges, len(values))
            ax.plot(centers, density, lw=2.6, color=analysis.COLORS[method], label=method)
            ax.fill_between(centers, density, color=analysis.COLORS[method], alpha=0.10)
        ax.axvline(0.0, ls="--", lw=2.0, color="black",
                   label=r"Optimal $\Delta G_{\mathrm{H}}$" if ei == 0 else None)
        ax.set_title(f"{element}\nBE shift={be_shifts[ei]:+.3f} eV", pad=9)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.tick_params(width=1.2, length=5)
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("Fraction-weighted ensemble density (eV$^{-1}$)")

    activity_text = " | ".join(f"{method}: {activities[method]:.6f}" for method in analysis.METHODS)
    fig.suptitle("FeCoNiPdPt fcc(100) bridge: average over all 19 temperatures\n"
                 fr"$\Delta G_{{\mathrm{{H}}}}$ = H BE - ({reference_hbe:+.3f} eV)"
                 "\nActivity (mean over 19 temperatures x 20 slabs; no natural log) | "
                 f"{activity_text}", fontsize=18, y=1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
