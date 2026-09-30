#!/usr/bin/env python3
"""Replot every facet/site probability-mass histogram for publication."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


FACET_SITE_ROOT = Path(
    "/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/"
    "new_BE_shift_completed_sites_log_KDE_peak/by_facet_site"
)
HISTOGRAM_DIRNAME = "probability_mass_weighted_histogram_1x5"
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")


def plot_case(output_dir: Path) -> Path:
    frame = pd.read_csv(output_dir / "weighted_histogram_bin_data.csv")
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    fig, axes = plt.subplots(1, 5, figsize=(25, 7.0), sharex=True, sharey=True)
    for ax, element in zip(axes, ELEMENTS):
        data = frame.loc[frame["element"] == element].sort_values("bin_index")
        centers = data["bin_center_eV"].to_numpy(float)
        width = float(data["bin_right_eV"].iloc[0] - data["bin_left_eV"].iloc[0])
        random_support = data["random_occupied"].astype(bool).to_numpy()

        ax.fill_between(
            centers, 0.0, 1.0, where=random_support, step="mid",
            transform=ax.get_xaxis_transform(), color="#777777", alpha=0.06,
        )
        ax.bar(
            centers, data["cemc_mass_inside_random_support"], width=0.92 * width,
            color="#2468B4", alpha=0.72, linewidth=0.0,
        )
        ax.bar(
            centers, data["cemc_mass_outside_random_support"], width=0.92 * width,
            color="#C44E52", alpha=0.82, linewidth=0.0,
        )
        ax.step(
            centers, data["random_fractional_probability_mass"], where="mid",
            color="#333333", lw=2.0,
        )
        ax.axvline(0.0, color="black", ls="--", lw=1.8)
        ax.set_title(element, pad=8)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.grid(alpha=0.18)

        mass_containment = float(data["cemc_mass_inside_random_support"].sum())
        ax.text(
            0.98, 0.95,
            fr"$C_{{\mathrm{{{element}}}}}$ = {100.0 * mass_containment:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=18,
        )

    axes[0].set_ylabel("Probability mass")
    legend = [
        Line2D([0], [0], color="#333333", lw=2.0,
               label="Random distribution"),
        Patch(facecolor="#2468B4", alpha=0.72,
              label="CEMC distribution inside Random"),
        Patch(facecolor="#C44E52", alpha=0.82,
              label="CEMC distribution outside Random"),
        Line2D([0], [0], color="black", ls="--", lw=1.8,
               label=r"Optimal $\Delta G_{\mathrm{H}}$"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_png = output_dir / "all_temperatures_probability_mass_weighted_histograms.png"
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    inputs = sorted(FACET_SITE_ROOT.glob(f"*/{HISTOGRAM_DIRNAME}/weighted_histogram_bin_data.csv"))
    if not inputs:
        raise FileNotFoundError("No weighted histogram CSV files found")
    for input_csv in inputs:
        output_png = plot_case(input_csv.parent)
        print(output_png, flush=True)


if __name__ == "__main__":
    main()
