#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


ARCHIVE = Path(
    "/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic/compact_cemc_10K/"
    "fcc111/cemc_structures_10K.npz"
)
OUTPUT = ARCHIVE.parent / "wc_vs_temperature_exact_attached_definition"
ELEMENT_INDEX = {"Pt": 0, "Pd": 1, "Ir": 2, "Rh": 3, "Ru": 4}
ATOMIC_NUMBER_TO_INDEX = {78: 0, 46: 1, 77: 2, 45: 3, 44: 4}
SELECTED_PAIRS = [
    ("Rh", "Pd"),
    ("Pd", "Pt"),
    ("Rh", "Pt"),
    ("Ir", "Pt"),
    ("Pd", "Pd"),
    ("Pt", "Pt"),
    ("Pd", "Ir"),
]


def surface_neighbor_graph(data):
    atoms = Atoms(
        numbers=data["Z"][0, 0],
        scaled_positions=data["scaled_positions"],
        cell=data["cell_A"],
        pbc=data["pbc"],
    )
    atoms.pbc = (True, True, False)
    z = np.round(atoms.positions[:, 2], 6)
    layers = [np.flatnonzero(z == value) for value in sorted(set(z))]
    if len(layers) != 10 or any(len(layer) != 100 for layer in layers):
        raise RuntimeError(f"Unexpected layers: {[len(layer) for layer in layers]}")
    surface_indices = layers[-1]
    surface = atoms[surface_indices]
    surface.pbc = (True, True, False)
    i0, j0, d0 = neighbor_list("ijd", surface, 5.0)
    mask = (i0 != j0) & (d0 > 1e-8)
    nearest = float(d0[mask].min())
    centers, neighbors, distances = neighbor_list("ijd", surface, nearest * 1.05)
    mask = (centers != neighbors) & (distances > 1e-8)
    centers, neighbors = centers[mask], neighbors[mask]
    degree = np.bincount(centers, minlength=len(surface_indices))
    if len(set(degree)) != 1:
        raise RuntimeError(f"Nonuniform surface degree: {sorted(set(degree))}")
    return surface_indices, centers, neighbors, int(degree[0]), nearest


def selected_wc_values(z, surface, centers, neighbors):
    types = np.asarray(
        [ATOMIC_NUMBER_TO_INDEX[int(number)] for number in z[surface]],
        dtype=np.int8,
    )
    populations = np.bincount(types, minlength=len(ELEMENT_INDEX)).astype(float)
    counts = np.zeros((len(ELEMENT_INDEX), len(ELEMENT_INDEX)), dtype=float)
    np.add.at(counts, (types[centers], types[neighbors]), 1.0)
    outgoing = counts.sum(axis=1)
    result = {}
    for center_name, neighbor_name in SELECTED_PAIRS:
        i = ELEMENT_INDEX[center_name]
        j = ELEMENT_INDEX[neighbor_name]
        p_ij = counts[i, j] / outgoing[i] if outgoing[i] > 0 else np.nan
        x_j = populations[j] / len(types)
        alpha = 1.0 - p_ij / x_j if x_j > 0 and np.isfinite(p_ij) else np.nan
        result[(center_name, neighbor_name)] = (alpha, p_ij, x_j)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(ARCHIVE) as data:
        temperatures = data["temperatures_K"].astype(int)
        z_all = data["Z"]
        surface, centers, neighbors, coordination, nearest = surface_neighbor_graph(data)
        series = {pair: [] for pair in SELECTED_PAIRS}
        rows = []
        for temperature_index, temperature in enumerate(temperatures):
            run_results = [
                selected_wc_values(z_all[temperature_index, run], surface, centers, neighbors)
                for run in range(z_all.shape[1])
            ]
            for pair in SELECTED_PAIRS:
                alpha = np.asarray([result[pair][0] for result in run_results])
                p_ij = np.asarray([result[pair][1] for result in run_results])
                x_j = np.asarray([result[pair][2] for result in run_results])
                valid = np.isfinite(alpha)
                mean = float(alpha[valid].mean()) if valid.any() else np.nan
                sd = float(alpha[valid].std(ddof=1)) if valid.sum() > 1 else np.nan
                series[pair].append(mean)
                rows.append([
                    int(temperature), f"{pair[0]}-{pair[1]}", mean, sd,
                    int(valid.sum()), float(np.nanmean(p_ij)), float(np.nanmean(x_j)),
                ])

    stem = "surface_WC_selected_pairs_exact_xj_10K"
    with (OUTPUT / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "temperature_K", "pair_i_j", "mean_alpha_ij", "sd_alpha_ij",
            "n_valid_slabs", "mean_P_ij", "mean_x_j_surface",
        ])
        writer.writerows(rows)

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    colors = plt.get_cmap("tab10")(np.arange(len(SELECTED_PAIRS)))
    fig, ax = plt.subplots(figsize=(13.0, 8.4), constrained_layout=True)
    for color, pair in zip(colors, SELECTED_PAIRS):
        ax.plot(
            temperatures,
            series[pair],
            color=color,
            linewidth=2.2,
            label=f"{pair[0]}-{pair[1]}",
        )
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
    ax.set_xlim(2000, 298)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 298])
    ax.set_xlabel("Temperature (K)", fontsize=24)
    ax.set_ylabel(r"$\alpha_{ij}^{\mathrm{surf}}$", fontsize=24)
    ax.tick_params(width=1.2, length=5, labelsize=18)
    ax.grid(False)
    ax.legend(
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        frameon=False,
        fontsize=18,
        handlelength=2.5,
        columnspacing=1.2,
    )
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "case": "PtPdIrRhRu fcc(111)",
        "source_archive": str(ARCHIVE),
        "temperature_count": len(temperatures),
        "n_slabs_per_temperature": int(z_all.shape[1]),
        "surface_atoms": len(surface),
        "surface_coordination": coordination,
        "surface_1NN_distance_A": nearest,
        "selected_directed_pairs_i_j": [f"{i}-{j}" for i, j in SELECTED_PAIRS],
        "wc_definition": "alpha_ij = 1 - P_ij/x_j",
        "P_ij": "probability of neighbor j in the surface 1NN shell of center i",
        "x_j": "N_j_surface/N_surface for each slab",
        "averaging": "alpha calculated per slab, then arithmetic mean over 20 slabs",
        "temperature_axis_direction": "2000 K left to 298 K right",
        "y_limits": [-1.0, 1.0],
        "style": {
            "title": False,
            "markers": False,
            "grid": False,
            "legend_frame": False,
            "base_font_size": 18,
            "axis_label_size": 24,
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PLOTTED {OUTPUT}")


if __name__ == "__main__":
    main()
