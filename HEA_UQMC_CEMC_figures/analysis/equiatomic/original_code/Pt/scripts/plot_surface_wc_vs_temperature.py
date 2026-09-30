#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


TEMPERATURES = (298, 300, *range(400, 2001, 100))


def surface_graph(template_path: Path):
    atoms = read(template_path)
    atoms.pbc = (True, True, False)
    z = np.round(atoms.positions[:, 2], 6)
    layers = [np.flatnonzero(z == value) for value in sorted(set(z))]
    if len(layers) != 10 or any(len(layer) != 100 for layer in layers):
        raise ValueError(f"{template_path}: unexpected layers {[len(x) for x in layers]}")
    surface_indices = layers[-1]
    surface = atoms[surface_indices]
    surface.pbc = (True, True, False)
    i0, j0, d0 = neighbor_list("ijd", surface, 5.0)
    mask = (i0 != j0) & (d0 > 1e-8)
    nearest = float(d0[mask].min())
    i, j, d = neighbor_list("ijd", surface, nearest * 1.05)
    mask = (i != j) & (d > 1e-8)
    i, j, d = i[mask], j[mask], d[mask]
    degree = np.bincount(i, minlength=len(surface_indices))
    if len(set(degree)) != 1:
        raise ValueError(f"{template_path}: nonuniform surface degree {sorted(set(degree))}")
    return surface_indices, i, j, int(degree[0]), nearest


def load_records(results: Path, temperature: int, z_to_index: dict[int, int]):
    path = results / "structures_by_temperature" / f"T{temperature:05d}" / "comp_0000.jsonl"
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[int(row["run"])] = np.asarray(
            [z_to_index[int(value)] for value in row["Z"]], dtype=np.int8
        )
    if sorted(records) != list(range(20)):
        raise ValueError(f"{path}: missing runs {sorted(records)}")
    return list(records.values())


def wc_structure(occ, surface, center_sites, neighbor_sites, n_elements):
    types = occ[surface]
    populations = np.bincount(types, minlength=n_elements).astype(float)
    pair_counts = np.zeros((n_elements, n_elements), float)
    np.add.at(pair_counts, (types[center_sites], types[neighbor_sites]), 1.0)
    outgoing = pair_counts.sum(axis=1)
    wc = np.full((n_elements, n_elements), np.nan)
    for i in range(n_elements):
        if outgoing[i] <= 0:
            continue
        for j in range(n_elements):
            reference = (populations[j] - float(i == j)) / (len(types) - 1)
            if reference > 0:
                wc[i, j] = 1.0 - (pair_counts[i, j] / outgoing[i]) / reference
    return wc


def analyze_case(
    label: str,
    elements: tuple[str, ...],
    z_to_index: dict[int, int],
    template: Path,
    results: Path,
    output: Path,
    excluded_pairs: set[frozenset[str]],
):
    output.mkdir(parents=True, exist_ok=True)
    surface, centers, neighbors, coordination, nearest = surface_graph(template)
    pairs = [
        (i, j)
        for i in range(len(elements))
        for j in range(i + 1, len(elements))
        if frozenset((elements[i], elements[j])) not in excluded_pairs
    ]
    series = {pair: [] for pair in pairs}
    rows = []
    for temperature in TEMPERATURES:
        matrices = [
            wc_structure(occ, surface, centers, neighbors, len(elements))
            for occ in load_records(results, temperature, z_to_index)
        ]
        stack = np.stack(matrices)
        for i, j in pairs:
            values = stack[:, i, j]
            valid = values[np.isfinite(values)]
            mean = float(np.mean(valid)) if len(valid) else np.nan
            sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
            series[(i, j)].append(mean)
            rows.append([temperature, f"{elements[i]}-{elements[j]}", mean, sd, len(valid)])

    stem = "surface_inplane_1NN_WC_vs_temperature"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_K", "pair", "mean_WC_alpha", "sd_WC_alpha", "n_valid_slabs"])
        writer.writerows(rows)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(pairs)))
    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    temperatures = np.asarray(TEMPERATURES)
    for color, (i, j) in zip(colors, pairs):
        ax.plot(
            temperatures,
            series[(i, j)],
            color=color,
            linewidth=2.0,
            label=f"{elements[i]}-{elements[j]}",
        )
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
    ax.set_xlim(2000, 298)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 298])
    ax.set_xlabel("Temperature (K)", fontsize=18)
    ax.set_ylabel(r"$(\alpha_{ij})^{\mathrm{surf}}$", fontsize=21)
    ax.tick_params(labelsize=14, width=1.2, length=5)
    ax.grid(alpha=0.20, linewidth=0.65)
    ax.legend(
        ncol=2,
        fontsize=10.5,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        frameon=True,
        framealpha=0.82,
        edgecolor="0.75",
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "case": label,
        "temperatures_K": list(TEMPERATURES),
        "temperature_axis_direction": "2000 K left to 298 K right",
        "n_slabs_per_temperature": 20,
        "pairs": [f"{elements[i]}-{elements[j]}" for i, j in pairs],
        "excluded_pairs": ["-".join(sorted(pair)) for pair in sorted(excluded_pairs, key=lambda x: sorted(x))],
        "surface_coordination": coordination,
        "surface_1NN_distance_A": nearest,
        "wc_definition": "1 - P(j|i)/((N_j-delta_ij)/(N_surface-1))",
        "y_limits": [-1.0, 1.0],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"DONE {label}: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-base", type=Path, required=True)
    parser.add_argument("--fe-base", type=Path, required=True)
    args = parser.parse_args()

    analyze_case(
        "PtPdRhRuIr fcc(111)",
        ("Pt", "Pd", "Rh", "Ru", "Ir"),
        {78: 0, 46: 1, 45: 2, 44: 3, 77: 4},
        args.pt_base / "inputs/template_fcc111_10x10x10.cif",
        args.pt_base / "results",
        args.pt_base / "surface_inplane_wc_vs_temperature",
        {
            frozenset(("Rh", "Ru")),
            frozenset(("Ru", "Ir")),
            frozenset(("Pt", "Ru")),
        },
    )

    fe_elements = ("Fe", "Co", "Ni", "Pd", "Pt")
    fe_z = {26: 0, 27: 1, 28: 2, 46: 3, 78: 4}
    for facet, site in (
        ("100", "bridge"),
        ("100", "hollow"),
        ("110", "bridge"),
        ("111", "hollow"),
        ("111", "top"),
    ):
        workdir = args.fe_base / facet / site
        analyze_case(
            f"FeCoNiPdPt fcc({facet}) {site}",
            fe_elements,
            fe_z,
            workdir / "inputs/template.cif",
            workdir / "results",
            workdir / "surface_inplane_wc_vs_temperature",
            set(),
        )


if __name__ == "__main__":
    main()
