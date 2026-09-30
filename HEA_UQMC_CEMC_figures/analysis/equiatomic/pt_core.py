#!/usr/bin/env python3
"""Element-resolved OH binding-energy distributions for PtPdIrRhRu slabs."""

from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read


BASE = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
MODEL_PATH = BASE / "inputs/activity_model_oh.txt"
TEMPLATE_PATH = BASE / "inputs/template_fcc111_10x10x10.cif"
ELEMENTS = ("Pt", "Pd", "Ir", "Rh", "Ru")
METHODS = ("Random", "CEMC", "CEMC + layer shuffle")
TEMPERATURES = tuple(range(2000, 299, -100)) + (298,)
KB_EV_PER_K = 8.617333262145e-5
ELEMENT_COLORS = {
    "Pt": "#CCB974", "Pd": "#8172B2", "Ir": "#4C72B0",
    "Rh": "#55A868", "Ru": "#C44E52",
}
Z_TO_ELEMENT = {78: "Pt", 46: "Pd", 77: "Ir", 45: "Rh", 44: "Ru"}
RECONSTRUCT_EXE = BASE / "bin/reconstruct_random_slab"
SELECTED_BE_SHIFTS_EV = {
    "Ir": -0.09, "Pd": -0.39, "Pt": 0.09, "Rh": -0.17, "Ru": -0.17,
}


def parse_model(path: Path):
    tokens = path.read_text().split()
    keys = {
        "ACTIVITY_MODEL_OH_V1", "elements", "intercept", "zone1", "zone2", "zone3",
        "e_opt", "activity_temperature", "zone2_ptr", "zone2_indices",
        "zone3_ptr", "zone3_indices",
    }
    sections = {}
    current = None
    for token in tokens:
        if token in keys:
            current = token
            sections[current] = []
        elif current is not None:
            sections[current].append(token)
    model_elements = sections["elements"]
    if set(model_elements) != set(ELEMENTS):
        raise ValueError(f"Unexpected model elements: {model_elements}")
    coeff = {}
    for zone in ("zone1", "zone2", "zone3"):
        raw = np.asarray(sections[zone], float)
        coeff[zone] = {element: float(raw[model_elements.index(element)]) for element in ELEMENTS}
    ptr = {zone: np.asarray(sections[f"zone{zone}_ptr"], int) for zone in (2, 3)}
    indices = {}
    for zone in (2, 3):
        raw = np.asarray(sections[f"zone{zone}_indices"], int)
        expected = int(raw[0])
        indices[zone] = raw[1:]
        if len(indices[zone]) != expected:
            raise ValueError(f"zone{zone} index count: {len(indices[zone])} != {expected}")
    if len(ptr[2]) != len(ptr[3]):
        raise ValueError("zone2/zone3 site counts differ")
    return {
        "intercept": float(sections["intercept"][0]),
        "e_opt": float(sections["e_opt"][0]),
        "activity_temperature": float(sections["activity_temperature"][0]),
        "coeff": coeff, "ptr": ptr, "indices": indices,
        "n_sites": len(ptr[2]) - 1, "model_elements": model_elements,
    }


def surface_sites_and_validate(model):
    atoms = read(TEMPLATE_PATH)
    atoms.pbc = (True, True, False)
    z = np.round(atoms.positions[:, 2], 6)
    layers = [np.flatnonzero(z == value) for value in sorted(set(z))]
    if len(layers) != 10 or any(len(layer) != 100 for layer in layers):
        raise ValueError(f"Unexpected layer sizes: {[len(layer) for layer in layers]}")
    surface = np.asarray(layers[-1], int)
    subsurface = set(map(int, layers[-2]))
    if len(surface) != model["n_sites"]:
        raise ValueError("OH site count does not equal surface atom count")
    if not np.array_equal(surface, np.arange(900, 1000)):
        raise ValueError(f"Expected CE surface site indices 900..999, found {surface.tolist()}")
    surface_set = set(map(int, surface))
    for site, center in enumerate(surface):
        z2 = model["indices"][2][model["ptr"][2][site]:model["ptr"][2][site + 1]]
        z3 = model["indices"][3][model["ptr"][3][site]:model["ptr"][3][site + 1]]
        if len(z2) != 6 or len(z3) != 3:
            raise ValueError(f"Site {site}: expected 6 zone2 and 3 zone3 atoms")
        if not set(map(int, z2)).issubset(surface_set):
            raise ValueError(f"Site {site}: zone2 is not entirely in the surface layer")
        if not set(map(int, z3)).issubset(subsurface):
            raise ValueError(f"Site {site}: zone3 is not entirely in the subsurface layer")
    return surface, layers


def occupancy_from_numbers(numbers):
    try:
        return np.asarray([Z_TO_ELEMENT[int(z)] for z in numbers], dtype="U2")
    except KeyError as exc:
        raise ValueError(f"Unexpected atomic number {exc.args[0]}") from exc


def reconstruct_random(run: int, tmpdir: Path):
    destination = tmpdir / f"initial_{run:02d}.json"
    command = [
        str(RECONSTRUCT_EXE), "--ce-export", str(BASE / "inputs/ce_export.txt"),
        "--seeds", str(BASE / "results/random_slab_seeds.csv"),
        "--trial", "0", "--composition-index", "0", "--run", str(run),
        "--output", str(destination),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return occupancy_from_numbers(json.loads(destination.read_text()))


def load_cemc(temperature: int):
    path = BASE / "results/structures_by_temperature" / f"T{temperature:05d}" / "comp_0000.jsonl"
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["run"])] = occupancy_from_numbers(row["Z"])
    if sorted(records) != list(range(20)):
        raise ValueError(f"Missing runs in {path}: {sorted(records)}")
    return records


def stable_seed(*parts):
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def shuffle_within_layers(occ, layers, seed):
    shuffled = occ.copy()
    rng = np.random.default_rng(seed)
    for layer in layers:
        values = shuffled[layer].copy()
        rng.shuffle(values)
        shuffled[layer] = values
    return shuffled


def evaluate(model, surface, occ, be_shifts):
    values = np.empty(model["n_sites"], float)
    center_elements = occ[surface]
    for site, center in enumerate(surface):
        element = occ[center]
        energy = model["intercept"] + model["coeff"]["zone1"][element]
        for zone in (2, 3):
            sites = model["indices"][zone][model["ptr"][zone][site]:model["ptr"][zone][site + 1]]
            energy += sum(model["coeff"][f"zone{zone}"][neighbor] for neighbor in occ[sites])
        energy -= be_shifts[element]
        values[site] = energy
    return values, center_elements


def smooth_density(values, element_labels, element, edges):
    selected = values[element_labels == element]
    counts, _ = np.histogram(selected, bins=edges)
    width = edges[1] - edges[0]
    density = counts.astype(float) / (len(values) * width)
    x = np.arange(-6, 7)
    kernel = np.exp(-0.5 * (x / 1.5) ** 2)
    kernel /= kernel.sum()
    return np.convolve(density, kernel, mode="same"), selected


def activity_no_log(values, optimum, temperature):
    return float(np.mean(np.exp(-np.abs(values - optimum) / (KB_EV_PER_K * temperature))))


def plot_distribution(packed, activities, title, output, optimum, stats_rows, temperature_scope,
                      be_shift_applied, be_shifts):
    all_values = np.concatenate([packed[method][0] for method in METHODS])
    lo = min(float(all_values.min()), optimum) - 0.08
    hi = max(float(all_values.max()), optimum) + 0.08
    edges = np.linspace(lo, hi, 101)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4), sharex=True, sharey=True)
    for method, ax in zip(METHODS, axes):
        values, labels = packed[method]
        for element in ELEMENTS:
            density, selected = smooth_density(values, labels, element, edges)
            ax.plot(centers, density, lw=2.6, color=ELEMENT_COLORS[element], label=element)
            stats_rows.append({
                "temperature_scope": temperature_scope, "method": method, "element": element,
                "n_all_OH_sites": len(values), "n_element_OH_sites": len(selected),
                "surface_fraction": len(selected) / len(values),
                "mean_OH_BE_eV": float(np.mean(selected)),
                "sd_OH_BE_eV": float(np.std(selected)),
                "median_OH_BE_eV": float(np.median(selected)),
                "optimal_OH_BE_eV": optimum, "BE_shift_applied": be_shift_applied,
                "selected_BE_shift_eV": be_shifts[element],
            })
        ax.axvline(optimum, color="black", ls="--", lw=2.2, label="Optimal OH BE (1.1 eV)")
        activity_values = np.asarray(activities[method], float)
        ax.set_title(f"{method}\nActivity = {activity_values.mean():.6f} ± {activity_values.std():.6f}",
                     pad=9)
        ax.set_xlabel("OH BE (eV)")
        ax.grid(alpha=0.20)
        ax.tick_params(width=1.2, length=5)
    axes[0].set_ylabel("Surface-fraction-weighted site density (eV$^{-1}$)")
    fig.suptitle(title, fontsize=18, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
               ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-be-shift", action="store_true")
    args = parser.parse_args()
    output = BASE / ("oh_be_distribution_analysis_be_shift" if args.apply_be_shift
                     else "oh_be_distribution_analysis_no_be_shift")
    plots = output / "plots_by_temperature"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    be_shifts = SELECTED_BE_SHIFTS_EV if args.apply_be_shift else {element: 0.0 for element in ELEMENTS}
    model = parse_model(MODEL_PATH)
    surface, layers = surface_sites_and_validate(model)
    if abs(model["e_opt"] - 1.1) > 1.0e-12:
        raise ValueError(f"Model optimum is {model['e_opt']}, expected 1.1 eV")
    by_temperature = {}
    activity_by_temperature = {}
    activity_rows = []
    rows = []
    with tempfile.TemporaryDirectory(prefix="ptpdirrhru_oh_") as tmp:
        tmpdir = Path(tmp)
        random_evaluated = [evaluate(model, surface, reconstruct_random(run, tmpdir), be_shifts)
                            for run in range(20)]
        for temperature in TEMPERATURES:
            cemc = load_cemc(temperature)
            evaluated = {
                "Random": random_evaluated,
                "CEMC": [evaluate(model, surface, cemc[run], be_shifts) for run in range(20)],
                "CEMC + layer shuffle": [
                    evaluate(model, surface, shuffle_within_layers(
                        cemc[run], layers,
                        stable_seed("PtPdRhRuIr-equi-atomic-layer-shuffle-v1", run, temperature)),
                             be_shifts)
                    for run in range(20)
                ],
            }
            packed = {
                method: (np.concatenate([item[0] for item in evaluated[method]]),
                         np.concatenate([item[1] for item in evaluated[method]]))
                for method in METHODS
            }
            by_temperature[temperature] = packed
            activities = {
                method: np.asarray([
                    activity_no_log(item[0], model["e_opt"], model["activity_temperature"])
                    for item in evaluated[method]
                ])
                for method in METHODS
            }
            activity_by_temperature[temperature] = activities
            for method in METHODS:
                for run, activity in enumerate(activities[method]):
                    activity_rows.append({
                        "temperature_K": temperature, "method": method, "slab": run + 1,
                        "activity_no_log": float(activity),
                        "n_OH_sites": model["n_sites"],
                        "activity_temperature_K": model["activity_temperature"],
                    })
            label = "final 298 K" if temperature == 298 else f"{temperature} K"
            stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
            plot_distribution(
                packed, activities,
                f"PtPdIrRhRu equi-atomic fcc(111): {label}\nOH binding-energy distribution",
                plots / f"{stem}_element_OH_BE_distribution.png",
                model["e_opt"], rows, str(temperature), args.apply_be_shift, be_shifts,
            )
    packed_all = {}
    for method in METHODS:
        packed_all[method] = (
            np.concatenate([by_temperature[t][method][0] for t in TEMPERATURES]),
            np.concatenate([by_temperature[t][method][1] for t in TEMPERATURES]),
        )
    average_rows = []
    activities_all = {
        method: np.asarray([
            np.mean([activity_by_temperature[t][method][run] for t in TEMPERATURES])
            for run in range(20)
        ])
        for method in METHODS
    }
    plot_distribution(
        packed_all, activities_all,
        "PtPdIrRhRu equi-atomic fcc(111): average over all 19 temperatures\n"
        "OH binding-energy distribution",
        plots / "all_temperatures_average_element_OH_BE_distribution.png",
        model["e_opt"], average_rows, "all_19_temperatures_equal_weight",
        args.apply_be_shift, be_shifts,
    )
    pd.DataFrame(rows).to_csv(output / "element_OH_BE_distribution_summary.csv", index=False)
    pd.DataFrame(average_rows).to_csv(
        output / "all_temperatures_average_element_OH_BE_summary.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(output / "activity_by_slab_temperature.csv", index=False)
    activity_average_rows = []
    for method in METHODS:
        for run, activity in enumerate(activities_all[method]):
            activity_average_rows.append({
                "temperature_scope": "mean_of_19_temperatures_per_slab",
                "method": method, "slab": run + 1, "activity_no_log": float(activity),
                "n_temperatures": len(TEMPERATURES),
                "activity_temperature_K": model["activity_temperature"],
            })
    pd.DataFrame(activity_average_rows).to_csv(
        output / "activity_by_slab_all_temperatures.csv", index=False)
    metadata = {
        "system": "PtPdIrRhRu", "facet": "fcc(111)", "composition": "equi-atomic",
        "methods": list(METHODS), "temperatures_K": list(TEMPERATURES),
        "n_independent_slabs": 20, "n_OH_top_sites_per_slab": model["n_sites"],
        "OH_BE_formula": "linear OH model: zone1 top atom + 6 surface 1NN + 3 subsurface atoms",
        "BE_shift_applied": args.apply_be_shift,
        "selected_BE_shifts_eV": be_shifts,
        "OH_BE_after_shift_formula": "linear_OH_BE - selected_BE_shift(center top-site element)",
        "optimal_OH_BE_eV": model["e_opt"],
        "activity_formula_no_log": "mean(exp(-abs(OH_BE - 1.1 eV)/(k_B*298 K))) over 100 OH sites per slab",
        "activity_plot_statistic": "mean ± population SD across 20 slab-level activities",
        "element_curve_normalization": "curve area equals the top-surface fraction of that element",
        "temperature_average": "equal weight over 19 saved temperatures and all 20 slabs",
        "n_png": len(TEMPERATURES) + 1, "n_pdf": 0,
        "model_elements": model["model_elements"], "plot_element_order": list(ELEMENTS),
        "surface_atoms": len(surface), "layer_sizes": [len(layer) for layer in layers],
        "structure_source": "CEMC JSONL plus deterministic reconstruction/shuffle; identical CE-site occupancies to saved CIFs",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
