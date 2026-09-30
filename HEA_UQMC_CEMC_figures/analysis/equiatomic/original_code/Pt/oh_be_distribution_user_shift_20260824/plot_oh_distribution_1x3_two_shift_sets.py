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
SHIFT_SETS = {
    "no_be_shift": {element: 0.0 for element in analysis.ELEMENTS},
    "specified_be_shift": {
        "Ir": 0.05068,
        "Pd": 0.53799,
        "Pt": 0.28892,
        "Rh": 0.20160,
        "Ru": 0.20917,
    },
}
OUTPUTS = {
    "no_be_shift": OUT / "oh_be_distribution_1x3_no_be_shift.png",
    "specified_be_shift": OUT / "oh_be_distribution_1x3_specified_be_shift.png",
}


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


def load_occupancies():
    random_occ = []
    cemc_occ = []
    shuffled_occ = []
    for run in range(analysis.N_RUNS):
        slab_dir = analysis.SLABS / f"slab_{run + 1:02d}"
        random_occ.append(analysis.occupancy(slab_dir / "initial_random.cif"))
        for temperature in analysis.TEMPERATURES:
            cemc_occ.append(analysis.occupancy(analysis.cemc_path(run, temperature)))
            shuffled_name = (
                "final_0298K_layer_shuffled.cif"
                if temperature == 298
                else f"T_{temperature:04d}K_layer_shuffled.cif"
            )
            shuffled_path = slab_dir / shuffled_name
            if not shuffled_path.is_file():
                raise FileNotFoundError(shuffled_path)
            shuffled_occ.append(analysis.occupancy(shuffled_path))
    return random_occ, cemc_occ, shuffled_occ


def evaluate_condition(model, surface, occupancies, shifts):
    random_occ, cemc_occ, shuffled_occ = occupancies
    return {
        "Homogeneous": concatenate(
            [analysis.common.evaluate(model, surface, occ, shifts) for occ in random_occ]
        ),
        "CEMC": concatenate(
            [analysis.common.evaluate(model, surface, occ, shifts) for occ in cemc_occ]
        ),
        "CEMC + layer shuffled": concatenate(
            [analysis.common.evaluate(model, surface, occ, shifts) for occ in shuffled_occ]
        ),
    }


def common_edges(results):
    metadata = json.loads((OUT / "analysis_metadata.json").read_text())
    reference = np.asarray(metadata["common_bin_edges_eV"], dtype=float)
    width = float(reference[1] - reference[0])
    all_values = np.concatenate(
        [
            values
            for condition in results.values()
            for values, _ in condition.values()
        ]
        + [np.asarray([analysis.OPTIMUM])]
    )
    needed_lo = float(all_values.min()) - 0.06
    needed_hi = float(all_values.max()) + 0.06
    lo_steps = min(0, int(np.floor((needed_lo - reference[0]) / width)))
    hi_steps = max(0, int(np.ceil((needed_hi - reference[-1]) / width)))
    lo = reference[0] + lo_steps * width
    hi = reference[-1] + hi_steps * width
    n_bins = int(round((hi - lo) / width))
    return np.linspace(lo, hi, n_bins + 1)


def draw(condition_name, packed, shifts, edges, model):
    centers = (edges[:-1] + edges[1:]) / 2.0
    activities = {
        method: analysis.common.activity_no_log(
            values, analysis.OPTIMUM, model["activity_temperature"]
        )
        for method, (values, _) in packed.items()
    }

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.4), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        values, labels = packed[method]
        for element in analysis.ELEMENTS:
            density = analysis.smooth_density(values, labels, element, edges)
            ax.plot(centers, density, color=COLORS[element], lw=2.7, label=element)
        ax.axvline(analysis.OPTIMUM, color="black", lw=2.1, ls="--")
        ax.set_title(method, pad=10)
        ax.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
        ax.grid(False)
        ax.tick_params(direction="out", width=1.2, length=5)
        activity_x, activity_y = (
            (0.38, 0.68) if method == "Homogeneous" else (0.04, 0.95)
        )
        activity_text = (
            f"{activities[method]:.6f}"
            if activities[method] >= 1.0e-5
            else f"{activities[method]:.3e}"
        )
        ax.text(
            activity_x,
            activity_y,
            f"Activity = {activity_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=17,
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
    fig.savefig(OUTPUTS[condition_name], dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"CONDITION={condition_name}")
    print("SHIFTS=" + ",".join(f"{e}:{shifts[e]:.5f}" for e in analysis.ELEMENTS))
    for method in METHODS:
        values, labels = packed[method]
        print(
            f"{method}: n={len(values)}, activity={activities[method]:.9f}, "
            f"min={values.min():.6f}, max={values.max():.6f}, "
            f"elements={dict(zip(*np.unique(labels, return_counts=True)))}"
        )
    print(f"OUTPUT={OUTPUTS[condition_name]}")


def main() -> None:
    model = analysis.common.parse_model(analysis.common.MODEL_PATH)
    surface, _ = analysis.common.surface_sites_and_validate(model)
    occupancies = load_occupancies()
    results = {
        name: evaluate_condition(model, surface, occupancies, shifts)
        for name, shifts in SHIFT_SETS.items()
    }
    edges = common_edges(results)
    print(f"COMMON_EDGES={edges[0]:.6f},{edges[-1]:.6f}; BINS={len(edges)-1}")
    for name, shifts in SHIFT_SETS.items():
        draw(name, results[name], shifts, edges, model)


if __name__ == "__main__":
    main()
