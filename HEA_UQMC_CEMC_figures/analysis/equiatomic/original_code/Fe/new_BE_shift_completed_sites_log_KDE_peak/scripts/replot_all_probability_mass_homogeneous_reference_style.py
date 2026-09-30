#!/usr/bin/env python3
"""Match all probability-mass histograms to the fcc100-hollow reference style."""

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import pandas as pd


FACET_SITE_ROOT = Path(
    "/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/"
    "new_BE_shift_completed_sites_log_KDE_peak/by_facet_site"
)
HISTOGRAM_DIRNAME = "probability_mass_weighted_histogram_1x5"
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")


def plot_case(output_dir: Path) -> tuple[Path, Path]:
    input_csv = output_dir / "weighted_histogram_bin_data.csv"
    output_png = output_dir / "all_temperatures_probability_mass_weighted_histograms.png"
    backup_png = output_dir / "all_temperatures_probability_mass_weighted_histograms.png.bak_before_font24_20260821"
    font24_backup_png = output_dir / "all_temperatures_probability_mass_weighted_histograms.png.bak_font24_before_0p1_ticks_band_20260821"
    homogeneous_backup_png = output_dir / "all_temperatures_probability_mass_weighted_histograms.png.before_homogeneous_reference_style_20260826"
    if not backup_png.exists():
        shutil.copy2(output_png, backup_png)
    if not font24_backup_png.exists():
        shutil.copy2(output_png, font24_backup_png)
    if not homogeneous_backup_png.exists():
        shutil.copy2(output_png, homogeneous_backup_png)

    frame = pd.read_csv(input_csv)
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 32,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 5, figsize=(24.5, 5.8), sharex=True, sharey=True)
    for ax, element in zip(axes, ELEMENTS):
        data = frame.loc[frame["element"] == element].sort_values("bin_index")
        centers = data["bin_center_eV"].to_numpy(float)
        width = float(data["bin_right_eV"].iloc[0] - data["bin_left_eV"].iloc[0])
        random_support = data["random_occupied"].astype(bool).to_numpy()

        ax.axvspan(-0.1, 0.1, color="#808080", alpha=0.15, zorder=0)
        ax.bar(
            centers, data["cemc_mass_inside_random_support"], width=0.92 * width,
            color="#4C78A8", alpha=0.80, linewidth=0.0,
        )
        ax.bar(
            centers, data["cemc_mass_outside_random_support"], width=0.92 * width,
            color="#E45756", alpha=0.82, linewidth=0.0,
        )
        ax.step(
            centers, data["random_fractional_probability_mass"], where="mid",
            color="black", lw=2.0,
        )
        ax.axvline(0.0, color="black", ls="--", lw=1.9)
        ax.set_title(element, pad=8)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.18)

        mass_containment = float(data["cemc_mass_inside_random_support"].sum())
        ax.text(
            0.04, 0.94,
            f"$C_{{{element}}}$ = {100.0 * mass_containment:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=28,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5),
        )

    axes[0].set_ylabel("Probability mass")
    legend = [
        Line2D([0], [0], color="black", lw=2.0, label="Homogeneous"),
        Patch(facecolor="#4C78A8", alpha=0.80,
              label="CEMC inside Homogeneous distribution"),
        Patch(facecolor="#E45756", alpha=0.82,
              label="CEMC outside Homogeneous distribution"),
        Patch(facecolor="#808080", alpha=0.15,
              label="Optimal window"),
        Line2D([0], [0], color="black", ls="--", lw=1.9,
               label="Optimal"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=5, frameon=False, fontsize=24,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.1)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_png, backup_png, font24_backup_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        output_png, backup_png, font24_backup_png = plot_case(args.output_dir)
        print(output_png, flush=True)
        print(f"backup={backup_png}", flush=True)
        print(f"font24_backup={font24_backup_png}", flush=True)
        return
    inputs = sorted(FACET_SITE_ROOT.glob(f"*/{HISTOGRAM_DIRNAME}/weighted_histogram_bin_data.csv"))
    if len(inputs) != 5:
        raise RuntimeError(f"Expected 5 facet/site inputs, found {len(inputs)}")
    for input_csv in inputs:
        output_png, backup_png, font24_backup_png = plot_case(input_csv.parent)
        print(output_png, flush=True)
        print(f"backup={backup_png}", flush=True)
        print(f"font24_backup={font24_backup_png}", flush=True)


if __name__ == "__main__":
    main()
