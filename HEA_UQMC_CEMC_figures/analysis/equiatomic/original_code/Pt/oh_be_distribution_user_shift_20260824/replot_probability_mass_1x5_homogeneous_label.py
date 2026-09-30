#!/usr/bin/env python3
"""Replot the requested-shift 1x5 histogram with Homogeneous legend text."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np


OUT = Path(
    "/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic/"
    "oh_be_distribution_user_shift_20260824"
)
INPUT = OUT / "probability_mass_weighted_histograms_1x5_bin_data_requested_shift.csv"
OUTPUT = OUT / "probability_mass_weighted_histograms_1x5.png"
ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")
OPTIMUM = 1.1
WINDOW = (1.0, 1.2)


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_element = {
        element: [row for row in rows if row["element"] == element]
        for element in ELEMENTS
    }
    first = by_element[ELEMENTS[0]]
    edges = np.asarray(
        [float(first[0]["bin_left_eV"])]
        + [float(row["bin_right_eV"]) for row in first]
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 32,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(24.5, 5.8), sharex=True, sharey=True)
    for ax, element in zip(axes, ELEMENTS):
        element_rows = by_element[element]
        homogeneous_mass = np.asarray(
            [float(row["random_probability_mass"]) for row in element_rows]
        )
        inside = np.asarray(
            [float(row["cemc_inside_random_support_mass"]) for row in element_rows]
        )
        outside = np.asarray(
            [float(row["cemc_outside_random_support_mass"]) for row in element_rows]
        )
        containment = float(inside.sum())

        ax.axvspan(*WINDOW, color="#808080", alpha=0.15, zorder=0)
        ax.bar(
            centers,
            inside,
            width=width * 0.92,
            color="#4C78A8",
            alpha=0.80,
            linewidth=0,
            zorder=1,
        )
        ax.bar(
            centers,
            outside,
            width=width * 0.92,
            color="#E45756",
            alpha=0.82,
            linewidth=0,
            zorder=1,
        )
        ax.stairs(homogeneous_mass, edges, color="black", lw=2.0, zorder=3)
        ax.axvline(OPTIMUM, color="black", ls="--", lw=1.9, zorder=4)
        ax.set_title(element, pad=8)
        ax.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.18)
        ax.text(
            0.04,
            0.94,
            f"$C_{{{element}}}$ = {100 * containment:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=28,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5),
        )

    axes[0].set_ylabel("Probability mass")
    legend = [
        Line2D([0], [0], color="black", lw=2, label="Homogeneous"),
        Patch(
            facecolor="#4C78A8",
            alpha=0.8,
            label="CEMC inside Homogeneous distribution",
        ),
        Patch(
            facecolor="#E45756",
            alpha=0.82,
            label="CEMC outside Homogeneous distribution",
        ),
        Patch(facecolor="#808080", alpha=0.15, label="Optimal window"),
        Line2D([0], [0], color="black", lw=1.9, ls="--", label="Optimal"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=24,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.1)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
