from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BASE = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
OUT = BASE / "oh_be_distribution_user_shift_20260824"
sys.path.insert(0, str(BASE))
import analyze_oh_be_user_shifts as analysis  # noqa: E402


COLORS = {
    "Pd": "#FFD700",
    "Pt": "#1F77B4",
    "Rh": "#FF9900",
    "Ru": "#E31A1C",
    "Ir": "#2CA02C",
}
METHODS = ("Homogeneous", "CEMC", "CEMC + layer shuffled")
OUTPUT = OUT / "oh_be_distribution_1x3_random_cemc_layer_shuffled.png"


def format_sig_figs(value: float, digits: int = 2) -> str:
    """Format with exactly ``digits`` significant figures, including zeros."""
    if value == 0:
        return f"{value:.{digits - 1}f}"
    decimal_places = digits - int(np.floor(np.log10(abs(value)))) - 1
    if decimal_places > 0:
        return f"{value:.{decimal_places}f}"
    return f"{value:.0f}"


def concatenate(packs):
    return (
        np.concatenate([pack[0] for pack in packs]),
        np.concatenate([pack[1] for pack in packs]),
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
        }
    )


def main() -> None:
    model = analysis.common.parse_model(analysis.common.MODEL_PATH)
    surface, _ = analysis.common.surface_sites_and_validate(model)

    random_packs = []
    cemc_packs = []
    shuffled_packs = []
    for run in range(analysis.N_RUNS):
        random_path = analysis.SLABS / f"slab_{run + 1:02d}" / "initial_random.cif"
        random_occ = analysis.occupancy(random_path)
        random_packs.append(
            analysis.common.evaluate(model, surface, random_occ, analysis.SHIFTS)
        )

        for temperature in analysis.TEMPERATURES:
            cemc_occ = analysis.occupancy(analysis.cemc_path(run, temperature))
            cemc_packs.append(
                analysis.common.evaluate(model, surface, cemc_occ, analysis.SHIFTS)
            )
            slab_dir = analysis.SLABS / f"slab_{run + 1:02d}"
            shuffled_name = (
                "final_0298K_layer_shuffled.cif"
                if temperature == 298
                else f"T_{temperature:04d}K_layer_shuffled.cif"
            )
            shuffled_path = slab_dir / shuffled_name
            if not shuffled_path.is_file():
                raise FileNotFoundError(shuffled_path)
            shuffled_occ = analysis.occupancy(shuffled_path)
            shuffled_packs.append(
                analysis.common.evaluate(model, surface, shuffled_occ, analysis.SHIFTS)
            )

    packed = {
        "Homogeneous": concatenate(random_packs),
        "CEMC": concatenate(cemc_packs),
        "CEMC + layer shuffled": concatenate(shuffled_packs),
    }

    metadata = json.loads((OUT / "analysis_metadata.json").read_text())
    edges = np.asarray(metadata["common_bin_edges_eV"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    activities = {
        method: analysis.common.activity_no_log(
            values, analysis.OPTIMUM, model["activity_temperature"]
        )
        for method, (values, _) in packed.items()
    }

    all_values = np.concatenate([values for values, _ in packed.values()])
    if all_values.min() < edges[0] or all_values.max() > edges[-1]:
        raise ValueError(
            f"Layer-shuffled values [{all_values.min()}, {all_values.max()}] "
            f"exceed existing histogram edges [{edges[0]}, {edges[-1]}]"
        )

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.4), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        values, labels = packed[method]
        for element in analysis.ELEMENTS:
            density = analysis.smooth_density(values, labels, element, edges)
            ax.plot(
                centers,
                density,
                color=COLORS[element],
                lw=2.7,
                label=element,
            )
        ax.axvline(analysis.OPTIMUM, color="black", lw=2.1, ls="--")
        ax.set_title(method, pad=10)
        ax.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
        ax.grid(False)
        ax.tick_params(direction="out", width=1.2, length=5)

        if method == "Homogeneous":
            # Keep the larger activity annotation clear of the optimum line.
            activity_x, activity_y = 0.0, 0.68
        else:
            activity_x, activity_y = 0.04, 0.95
        ax.text(
            activity_x,
            activity_y,
            f"Activity = {format_sig_figs(activities[method])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=24,
            fontstretch="condensed",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3.0),
        )

    axes[0].set_ylabel("Probability density")
    handles = [
        Line2D([0], [0], color=COLORS[element], lw=2.7, label=element)
        for element in analysis.ELEMENTS
    ]
    legend = axes[0].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        columnspacing=1.4,
        handlelength=2.4,
        fontsize=18,
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for method in METHODS:
        values, labels = packed[method]
        print(
            f"{method}: n={len(values)}, activity={activities[method]:.9f}, "
            f"min={values.min():.6f}, max={values.max():.6f}, "
            f"elements={dict(zip(*np.unique(labels, return_counts=True)))}"
        )
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
