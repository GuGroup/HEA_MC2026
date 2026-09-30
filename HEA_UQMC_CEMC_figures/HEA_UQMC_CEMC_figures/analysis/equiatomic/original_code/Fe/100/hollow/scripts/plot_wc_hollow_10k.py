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


def surface_graph(data):
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
    i, j, d = neighbor_list("ijd", surface, nearest * 1.05)
    mask = (i != j) & (d > 1e-8)
    i, j, d = i[mask], j[mask], d[mask]
    degree = np.bincount(i, minlength=100)
    if len(set(degree)) != 1:
        raise RuntimeError(f"Nonuniform surface degree {sorted(set(degree))}")
    return surface_indices, i, j, int(degree[0]), nearest


def wc_for_structure(z, surface, centers, neighbors, atomic_number_to_index, n_elements):
    types = np.asarray([atomic_number_to_index[int(v)] for v in z[surface]], dtype=np.int8)
    populations = np.bincount(types, minlength=n_elements).astype(float)
    counts = np.zeros((n_elements, n_elements), dtype=float)
    np.add.at(counts, (types[centers], types[neighbors]), 1.0)
    outgoing = counts.sum(axis=1)
    wc = np.full((n_elements, n_elements), np.nan)
    for i in range(n_elements):
        if outgoing[i] <= 0:
            continue
        for j in range(n_elements):
            reference = populations[j] / len(types)
            if reference > 0:
                wc[i, j] = 1.0 - (counts[i, j] / outgoing[i]) / reference
    return wc


def plot_case(
    archive: Path,
    output: Path,
    elements: tuple[str, ...],
    atomic_number_to_index: dict[int, int],
    excluded_pairs: set[frozenset[str]],
    case_name: str,
):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(archive) as data:
        temperatures = data["temperatures_K"].astype(int)
        z_all = data["Z"]
        surface, centers, neighbors, coordination, nearest = surface_graph(data)
        pairs = [
            (i, j)
            for i in range(len(elements))
            for j in range(i + 1, len(elements))
            if frozenset((elements[i], elements[j])) not in excluded_pairs
        ]
        series = {pair: [] for pair in pairs}
        rows = []
        for tidx, temperature in enumerate(temperatures):
            stack = np.stack([
                wc_for_structure(
                    z_all[tidx, run], surface, centers, neighbors,
                    atomic_number_to_index, len(elements),
                )
                for run in range(z_all.shape[1])
            ])
            for i, j in pairs:
                values = stack[:, i, j]
                valid = values[np.isfinite(values)]
                mean = float(valid.mean()) if len(valid) else np.nan
                sd = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
                series[(i, j)].append(mean)
                rows.append([int(temperature), f"{elements[i]}-{elements[j]}", mean, sd, len(valid)])

    stem = "surface_inplane_1NN_WC_vs_temperature"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_K", "pair", "mean_WC_alpha", "sd_WC_alpha", "n_valid_slabs"])
        writer.writerows(rows)

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(pairs)))
    fig, ax = plt.subplots(figsize=(13.0, 8.4), constrained_layout=True)
    for color, (i, j) in zip(colors, pairs):
        ax.plot(
            temperatures,
            series[(i, j)],
            color=color,
            linewidth=2.2,
            label=f"{elements[i]}-{elements[j]}",
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
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "case": case_name,
        "source_archive": str(archive),
        "temperature_count": len(temperatures),
        "temperature_axis_direction": "2000 K left to 298 K right",
        "n_slabs_per_temperature": 20,
        "pairs": [f"{elements[i]}-{elements[j]}" for i, j in pairs],
        "excluded_pairs": ["-".join(sorted(pair)) for pair in sorted(excluded_pairs, key=lambda x: sorted(x))],
        "surface_coordination": coordination,
        "surface_1NN_distance_A": nearest,
        "wc_definition": "alpha_ij = 1 - P_ij/x_j",
        "x_j": "N_j_surface/N_surface for each slab",
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
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PLOTTED {case_name}: {output}", flush=True)


def main():
    fe_root = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/100/hollow")
    plot_case(
        fe_root / "compact_cemc_10K/cemc_structures_10K.npz",
        fe_root / "surface_inplane_wc_vs_temperature",
        ("Fe", "Co", "Ni", "Pd", "Pt"),
        {26: 0, 27: 1, 28: 2, 46: 3, 78: 4},
        set(),
        "FeCoNiPdPt fcc(100) hollow",
    )


if __name__ == "__main__":
    main()
