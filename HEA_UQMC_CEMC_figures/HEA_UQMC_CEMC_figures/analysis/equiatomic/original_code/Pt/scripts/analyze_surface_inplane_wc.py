#!/usr/bin/env python3
"""Surface-layer in-plane 1NN Warren-Cowley analysis for PtPdIrRhRu."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list

import analyze_oh_be_distributions_no_shift as analysis


BASE = analysis.BASE
OUTPUT = BASE / "surface_inplane_wc_analysis"
ELEMENTS = analysis.ELEMENTS
METHODS = analysis.METHODS
ELEMENT_TO_INDEX = {element: i for i, element in enumerate(ELEMENTS)}


def surface_neighbors():
    atoms = read(analysis.TEMPLATE_PATH)
    atoms.pbc = (True, True, False)
    keys = np.round(atoms.positions[:, 2], 6)
    layers = [np.flatnonzero(keys == value) for value in sorted(set(keys))]
    if len(layers) != 10 or any(len(layer) != 100 for layer in layers):
        raise ValueError(f"Unexpected layer sizes: {[len(layer) for layer in layers]}")
    surface_indices = layers[-1]
    surface = atoms[surface_indices]
    surface.pbc = (True, True, False)
    i0, j0, d0 = neighbor_list("ijd", surface, 5.0)
    mask = (i0 != j0) & (d0 > 1.0e-8)
    distance = float(d0[mask].min())
    i, j, d = neighbor_list("ijd", surface, distance * 1.05)
    mask = (i != j) & (d > 1.0e-8)
    i, j, d = i[mask], j[mask], d[mask]
    degree = np.bincount(i, minlength=len(surface_indices))
    if not np.all(degree == 6):
        raise ValueError(f"fcc(111) surface in-plane degree: {sorted(set(degree))}")
    if float(np.max(np.abs(d - distance))) > 1.0e-5:
        raise ValueError("Nonuniform fcc(111) surface 1NN distance")
    return layers, surface_indices, i, j, distance


def wc_for_structure(occ, surface_indices, centers, neighbors):
    types = np.asarray([ELEMENT_TO_INDEX[element] for element in occ[surface_indices]], int)
    populations = np.bincount(types, minlength=5).astype(float)
    pair_counts = np.zeros((5, 5), float)
    np.add.at(pair_counts, (types[centers], types[neighbors]), 1.0)
    outgoing = pair_counts.sum(axis=1)
    wc = np.full((5, 5), np.nan)
    for i in range(5):
        if outgoing[i] <= 0:
            continue
        for j in range(5):
            reference = (populations[j] - float(i == j)) / (len(types) - 1)
            if reference > 0:
                wc[i, j] = 1.0 - (pair_counts[i, j] / outgoing[i]) / reference
    return wc, populations


def summarize(matrices):
    stack = np.stack(matrices)
    valid = np.isfinite(stack)
    n_valid = valid.sum(axis=0)
    mean = np.divide(np.nansum(stack, axis=0), n_valid,
                     out=np.full((5, 5), np.nan), where=n_valid > 0)
    centered = np.where(valid, stack - mean, 0.0)
    sd = np.sqrt(np.divide(np.sum(centered * centered, axis=0), np.maximum(n_valid - 1, 1)))
    sd[n_valid < 2] = np.nan
    ci95 = 1.96 * sd / np.sqrt(n_valid)
    return mean, sd, ci95, n_valid


def plot_heatmap(summaries, output):
    means = [summaries[method][0] for method in METHODS]
    vmax = max(float(np.nanmax(np.abs(matrix))) for matrix in means)
    vmax = max(vmax, 0.02)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.3), sharex=True, sharey=True)
    image = None
    for ax, method, matrix in zip(axes, METHODS, means):
        image = ax.imshow(matrix, cmap="coolwarm", norm=norm, interpolation="nearest")
        ax.set_title(method)
        ax.set_xticks(range(5), ELEMENTS)
        ax.set_yticks(range(5), ELEMENTS)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Neighbor element, j")
        for row in range(5):
            for col in range(5):
                value = matrix[row, col]
                label = "--" if not np.isfinite(value) else f"{value:+.3f}"
                color = "white" if np.isfinite(value) and abs(value) > 0.56 * vmax else "black"
                ax.text(col, row, label, ha="center", va="center", fontsize=14, color=color)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Center element, i")
    colorbar_axis = fig.add_axes([0.915, 0.19, 0.018, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"Warren--Cowley $\alpha_{ij}$", fontsize=18)
    colorbar.ax.tick_params(labelsize=15)
    fig.suptitle("PtPdIrRhRu equi-atomic fcc(111): surface-layer in-plane 1NN WC\n"
                 "Average over all 19 temperatures and 20 slabs", fontsize=18, y=1.02)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.83, wspace=0.16)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    layers, surface, centers, neighbors, distance = surface_neighbors()
    matrices = {method: [] for method in METHODS}
    populations = {method: [] for method in METHODS}
    with tempfile.TemporaryDirectory(prefix="ptpdirrhru_wc_") as tmp:
        tmpdir = Path(tmp)
        for run in range(20):
            wc, pop = wc_for_structure(analysis.reconstruct_random(run, tmpdir), surface, centers, neighbors)
            matrices["Random"].append(wc)
            populations["Random"].append(pop)
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc(temperature)
            for run in range(20):
                wc, pop = wc_for_structure(cemc[run], surface, centers, neighbors)
                matrices["CEMC"].append(wc)
                populations["CEMC"].append(pop)
                shuffled = analysis.shuffle_within_layers(
                    cemc[run], layers,
                    analysis.stable_seed("PtPdRhRuIr-equi-atomic-layer-shuffle-v1", run, temperature))
                wc, pop = wc_for_structure(shuffled, surface, centers, neighbors)
                matrices["CEMC + layer shuffle"].append(wc)
                populations["CEMC + layer shuffle"].append(pop)
    summaries = {method: summarize(matrices[method]) for method in METHODS}
    rows = []
    composition_rows = []
    for method in METHODS:
        mean, sd, ci95, n_valid = summaries[method]
        for i, center in enumerate(ELEMENTS):
            for j, neighbor in enumerate(ELEMENTS):
                rows.append({
                    "method": method, "center_element": center, "neighbor_element": neighbor,
                    "mean_WC_alpha": mean[i, j], "sd_WC_alpha": sd[i, j],
                    "ci95_half_width": ci95[i, j], "n_valid_structures": int(n_valid[i, j]),
                    "n_input_structures": len(matrices[method]),
                    "surface_layer_atoms": len(surface), "surface_inplane_1NN_coordination": 6,
                    "surface_inplane_1NN_distance_A": distance,
                })
        pop_stack = np.stack(populations[method])
        for i, element in enumerate(ELEMENTS):
            composition_rows.append({
                "method": method, "element": element,
                "mean_surface_atoms": float(pop_stack[:, i].mean()),
                "sd_surface_atoms": float(pop_stack[:, i].std()),
                "mean_surface_fraction": float(pop_stack[:, i].mean() / len(surface)),
                "n_structures": len(pop_stack),
            })
    pd.DataFrame(rows).to_csv(OUTPUT / "all_temperatures_surface_layer_inplane_1NN_WC.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(OUTPUT / "surface_composition_summary.csv", index=False)
    plot_heatmap(summaries, OUTPUT / "all_temperatures_surface_layer_inplane_1NN_WC_heatmap.png")
    metadata = {
        "system": "PtPdIrRhRu", "facet": "fcc(111)", "methods": list(METHODS),
        "temperatures_K": list(analysis.TEMPERATURES), "random_structures": 20,
        "cemc_structures": 380, "layer_shuffled_structures": 380,
        "surface_layer_atoms": 100, "surface_inplane_1NN_coordination": 6,
        "surface_inplane_1NN_distance_A": distance,
        "wc_definition": "1 - P(j|i)/((N_j-delta_ij)/(N_surface-1))",
        "normalization": "each structure's actual surface-layer composition; center atom excluded",
        "averaging": "structure-level WC followed by equal-weight mean over available finite values",
        "random_temperature_handling": "20 unique random slabs used once",
        "n_png": 1, "n_pdf": 0,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
