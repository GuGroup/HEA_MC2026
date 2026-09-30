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


def top_surface_indices(data: np.lib.npyio.NpzFile) -> np.ndarray:
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
    return layers[-1]


def plot_case(
    archive: Path,
    output: Path,
    elements: tuple[str, ...],
    atomic_numbers: tuple[int, ...],
    case_name: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(archive) as data:
        temperatures = data["temperatures_K"].astype(int)
        expected = set(range(300, 2001, 10)) | {298}
        if set(temperatures.tolist()) != expected:
            missing = sorted(expected - set(temperatures.tolist()))
            extra = sorted(set(temperatures.tolist()) - expected)
            raise RuntimeError(f"Unexpected temperatures; missing={missing}, extra={extra}")
        order = np.argsort(temperatures)[::-1]
        temperatures = temperatures[order]
        z_all = data["Z"][order]
        surface = top_surface_indices(data)

        means = np.zeros((len(temperatures), len(elements)), dtype=float)
        sds = np.zeros_like(means)
        rows = []
        for tidx, temperature in enumerate(temperatures):
            fractions = np.stack([
                np.asarray([
                    np.count_nonzero(z_all[tidx, run, surface] == atomic_number) / len(surface)
                    for atomic_number in atomic_numbers
                ], dtype=float)
                for run in range(z_all.shape[1])
            ])
            if not np.allclose(fractions.sum(axis=1), 1.0):
                raise RuntimeError(f"Surface fractions do not sum to one at {temperature} K")
            means[tidx] = fractions.mean(axis=0)
            sds[tidx] = fractions.std(axis=0, ddof=1)
            for eidx, element in enumerate(elements):
                rows.append([
                    int(temperature), element, means[tidx, eidx], sds[tidx, eidx],
                    fractions.shape[0], len(surface),
                ])

    stem = "surface_atomic_fraction_vs_temperature_10K"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "temperature_K", "element", "mean_surface_atomic_fraction",
            "sd_surface_atomic_fraction", "n_slabs", "surface_atoms_per_slab",
        ])
        writer.writerows(rows)

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(elements)))
    fig, ax = plt.subplots(figsize=(13.0, 8.4), constrained_layout=True)
    for eidx, (element, color) in enumerate(zip(elements, colors)):
        ax.plot(
            temperatures,
            means[:, eidx],
            color=color,
            linewidth=2.2,
            label=element,
        )
    ax.set_xlim(2000, 298)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 298])
    ax.set_xlabel("Temperature (K)", fontsize=24)
    ax.set_ylabel("Surface atomic fraction", fontsize=24)
    ax.tick_params(width=1.2, length=5, labelsize=18)
    ax.grid(False)
    ax.legend(
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
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
        "surface_layer": "topmost atomic layer",
        "surface_atoms_per_slab": int(len(surface)),
        "temperature_count": int(len(temperatures)),
        "temperature_spacing": "10 K from 2000 K through 300 K, plus 298 K",
        "temperature_axis_direction": "2000 K left to 298 K right",
        "n_slabs_per_temperature": int(z_all.shape[1]),
        "elements": list(elements),
        "quantity": "mean N_element_surface/N_surface over slabs",
        "y_limits": [0.0, 1.0],
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


def main() -> None:
    pt_root = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
    fe_root = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/100/hollow")
    plot_case(
        pt_root / "compact_cemc_10K/fcc111/cemc_structures_10K.npz",
        pt_root / "surface_atomic_fraction",
        ("Pt", "Pd", "Ir", "Rh", "Ru"),
        (78, 46, 77, 45, 44),
        "PtPdIrRhRu fcc(111)",
    )
    plot_case(
        fe_root / "compact_cemc_10K/cemc_structures_10K.npz",
        Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/surface_atomic_fraction/100"),
        ("Fe", "Co", "Ni", "Pd", "Pt"),
        (26, 27, 28, 46, 78),
        "FeCoNiPdPt fcc(100) hollow",
    )


if __name__ == "__main__":
    main()
