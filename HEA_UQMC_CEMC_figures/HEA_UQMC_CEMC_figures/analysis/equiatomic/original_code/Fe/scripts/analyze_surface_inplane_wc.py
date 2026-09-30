#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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

import analyze_hbe_distributions as analysis


BASE = analysis.BASE
ELEMENTS = analysis.ELEMENTS
METHODS = analysis.METHODS
EXPECTED_INPLANE_COORDINATION = {"100": 4, "110": 2, "111": 6}


def surface_inplane_neighbors(template_path: Path, facet: str):
    atoms = read(template_path)
    groups = analysis.layer_groups(template_path)
    surface_indices = groups[-1]
    surface = atoms[surface_indices]
    surface.pbc = (True, True, False)

    i0, j0, d0 = neighbor_list("ijd", surface, 5.0)
    mask = (i0 != j0) & (d0 > 1.0e-8)
    nearest_distance = float(d0[mask].min())
    i, j, distances = neighbor_list("ijd", surface, nearest_distance * 1.05)
    mask = (i != j) & (distances > 1.0e-8)
    i, j, distances = i[mask], j[mask], distances[mask]
    degree = np.bincount(i, minlength=len(surface_indices))
    expected = EXPECTED_INPLANE_COORDINATION[facet]
    if not np.all(degree == expected):
        raise ValueError(f"fcc({facet}) surface in-plane degree: {sorted(set(degree))}, expected {expected}")
    if np.max(np.abs(distances - nearest_distance)) > 1.0e-5:
        raise ValueError(f"fcc({facet}) nonuniform surface 1NN distances")
    return groups, surface_indices, i, j, nearest_distance, expected


def wc_for_structure(occ: np.ndarray, surface_indices: np.ndarray,
                     center_sites: np.ndarray, neighbor_sites: np.ndarray) -> np.ndarray:
    types = occ[surface_indices]
    n_surface = len(types)
    populations = np.bincount(types, minlength=len(ELEMENTS)).astype(float)
    pair_counts = np.zeros((len(ELEMENTS), len(ELEMENTS)), float)
    np.add.at(pair_counts, (types[center_sites], types[neighbor_sites]), 1.0)
    outgoing = pair_counts.sum(axis=1)
    wc = np.full_like(pair_counts, np.nan)
    for center in range(len(ELEMENTS)):
        if outgoing[center] <= 0.0:
            continue
        for neighbor in range(len(ELEMENTS)):
            # Finite-size-corrected random reference for a neighbor selected
            # from the same layer while excluding the central atom itself.
            reference = (populations[neighbor] - float(center == neighbor)) / (n_surface - 1)
            if reference > 0.0:
                probability = pair_counts[center, neighbor] / outgoing[center]
                wc[center, neighbor] = 1.0 - probability / reference
    return wc


def summarize(matrices: list[np.ndarray]):
    stack = np.stack(matrices)
    valid = np.isfinite(stack)
    n_valid = valid.sum(axis=0)
    mean = np.divide(np.nansum(stack, axis=0), n_valid,
                     out=np.full((5, 5), np.nan), where=n_valid > 0)
    centered = np.where(valid, stack - mean, 0.0)
    denominator = np.maximum(n_valid - 1, 1)
    sd = np.sqrt(np.sum(centered * centered, axis=0) / denominator)
    sd[n_valid < 2] = np.nan
    ci95 = 1.96 * sd / np.sqrt(n_valid)
    return mean, sd, ci95, n_valid


def plot_heatmaps(case: str, facet: str, site: str, summaries, output: Path):
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
    fig.suptitle(f"FeCoNiPdPt fcc({facet}) {site}: surface-layer in-plane 1NN WC\n"
                 "Average over all available temperatures and slabs", fontsize=18, y=1.02)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.83, wspace=0.16)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_case(facet: str, site: str, case: str):
    workdir = BASE / facet / site
    output_dir = workdir / "surface_inplane_wc_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, surface_indices, center_sites, neighbor_sites, distance, coordination = \
        surface_inplane_neighbors(workdir / "inputs/template.cif", facet)

    matrices = {method: [] for method in METHODS}
    with tempfile.TemporaryDirectory(prefix=f"surface_wc_{case}_") as tmp:
        tmpdir = Path(tmp)
        for run in range(20):
            occ = analysis.reconstruct_random(workdir, run, tmpdir)
            matrices["Random"].append(wc_for_structure(occ, surface_indices, center_sites, neighbor_sites))

        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(workdir, temperature)
            for run in range(20):
                occ = cemc[run]
                matrices["CEMC"].append(
                    wc_for_structure(occ, surface_indices, center_sites, neighbor_sites))
                seed = analysis.stable_seed("FeCoNiPdPt-equi-layer-shuffle-v1", f"{facet}/{site}", run, temperature)
                shuffled = analysis.shuffle_occ(occ, groups, seed)
                matrices["CEMC + layer shuffle"].append(
                    wc_for_structure(shuffled, surface_indices, center_sites, neighbor_sites))

    summaries = {method: summarize(matrices[method]) for method in METHODS}
    rows = []
    for method in METHODS:
        mean, sd, ci95, n_valid = summaries[method]
        for i, center in enumerate(ELEMENTS):
            for j, neighbor in enumerate(ELEMENTS):
                rows.append({
                    "facet": facet, "site": site, "method": method,
                    "center_element": center, "neighbor_element": neighbor,
                    "mean_WC_alpha": mean[i, j], "sd_WC_alpha": sd[i, j],
                    "ci95_half_width": ci95[i, j], "n_valid_structures": int(n_valid[i, j]),
                    "n_input_structures": len(matrices[method]),
                    "surface_layer_atoms": len(surface_indices),
                    "surface_inplane_1NN_coordination": coordination,
                    "surface_inplane_1NN_distance_A": distance,
                })
    table = pd.DataFrame(rows)
    csv_path = output_dir / "all_temperatures_surface_layer_inplane_1NN_WC.csv"
    png_path = output_dir / "all_temperatures_surface_layer_inplane_1NN_WC_heatmap.png"
    table.to_csv(csv_path, index=False)
    plot_heatmaps(case, facet, site, summaries, png_path)
    metadata = {
        "facet": facet, "site": site, "methods": list(METHODS),
        "temperature_snapshots_K": list(analysis.TEMPERATURES),
        "random_structures": 20, "cemc_structures": 20 * len(analysis.TEMPERATURES),
        "layer_shuffled_structures": 20 * len(analysis.TEMPERATURES),
        "surface_layer_atoms": int(len(surface_indices)),
        "surface_inplane_1NN_coordination": int(coordination),
        "surface_inplane_1NN_distance_A": distance,
        "wc_definition": "1 - P(j|i)/((N_j-delta_ij)/(N_surface-1))",
        "normalization": "each slab's actual surface-layer composition; central atom excluded",
        "random_temperature_handling": "20 unique random slabs used once because random slabs are temperature-independent",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return table, metadata


def main():
    tables = []
    metadata = []
    for facet, site, case in analysis.COMBOS:
        table, meta = analyze_case(facet, site, case)
        tables.append(table)
        metadata.append(meta)
        print(f"Ready {case}: {len(table)} WC rows")
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(BASE / "surface_layer_inplane_WC_all_facets_sites.csv", index=False)
    result = {
        "validated": True,
        "n_cases": len(analysis.COMBOS),
        "n_heatmaps_png": len(analysis.COMBOS),
        "n_pdf": 0,
        "n_combined_rows": len(combined),
        "cases": metadata,
    }
    (BASE / "surface_layer_inplane_WC_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
