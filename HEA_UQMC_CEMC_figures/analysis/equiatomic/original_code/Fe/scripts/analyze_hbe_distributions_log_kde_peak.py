#!/usr/bin/env python3
"""Element-resolved H binding-energy distributions for FeCoNiPdPt slab ensembles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
FIT_ROOT = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/fit_exp")
SHIFT_ROOT = FIT_ROOT / "log_activity_joint_probability_tsne_activity_maps"
RECONSTRUCT_EXE = BASE / "bin" / "reconstruct_random_slab"
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
ANALYSIS_1X3 = "hbe_distribution_analysis_log_activity_KDE_peak_trial_BE_shift_1x3"
OPTIMAL_DELTA_G_H_EV = 0.0
KB_EV_PER_K = 8.617333262145e-5


class NoPdfOutput:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def savefig(self, *args, **kwargs):
        pass
Z_TO_INDEX = {26: 0, 27: 1, 28: 2, 46: 3, 78: 4}
COMBOS = (
    ("100", "bridge", "fcc100_bridge"),
    ("100", "hollow", "fcc100_hollow"),
    ("110", "bridge", "fcc110_bridge"),
    ("111", "hollow", "fcc111_hollow"),
    ("111", "top", "fcc111_top"),
)
TEMPERATURES = tuple(range(2000, 299, -100)) + (298,)
METHODS = ("Random", "CEMC", "CEMC + layer shuffle")
COLORS = {"Random": "#555555", "CEMC": "#2468B4", "CEMC + layer shuffle": "#E68613"}
ELEMENT_COLORS = {
    "Fe": "#C44E52", "Co": "#4C72B0", "Ni": "#55A868",
    "Pd": "#8172B2", "Pt": "#CCB974",
}
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


@dataclass
class ActivityModel:
    site_type: str
    elements: list[str]
    intercept: float
    e_opt: float
    activity_temperature: float
    zone1_element: np.ndarray
    zone_coeff: dict[int, np.ndarray]
    zone1_combo: dict[str, float]
    zone3_combo: dict[str, float]
    ptr: dict[int, np.ndarray]
    indices: dict[int, np.ndarray]
    n_sites: int


def parse_model(path: Path) -> ActivityModel:
    tok = path.read_text().split()
    pos = 0
    if tok[pos] != "ACTIVITY_MODEL_H_V1":
        raise ValueError(f"Bad model magic: {path}")
    pos += 1
    site_type = ""
    elements: list[str] = []
    intercept = 0.0
    e_opt = math.nan
    activity_temperature = 298.0
    zone1_element = np.zeros(5)
    zone_coeff = {z: np.zeros(5) for z in range(2, 6)}
    zone1_combo: dict[str, float] = {}
    zone3_combo: dict[str, float] = {}
    ptr: dict[int, np.ndarray] = {}
    indices: dict[int, np.ndarray] = {}
    n_sites = -1
    while pos < len(tok):
        key = tok[pos]
        pos += 1
        if key == "site_type":
            site_type = tok[pos]; pos += 1
        elif key == "elements":
            elements = tok[pos:pos + 5]; pos += 5
        elif key in ("intercept", "e_opt", "activity_temperature"):
            value = float(tok[pos]); pos += 1
            if key == "intercept": intercept = value
            elif key == "e_opt": e_opt = value
            elif key == "activity_temperature": activity_temperature = value
        elif key == "zone1_element":
            zone1_element = np.asarray(tok[pos:pos + 5], float); pos += 5
        elif key in ("zone2", "zone3", "zone4", "zone5"):
            z = int(key[-1])
            zone_coeff[z] = np.asarray(tok[pos:pos + 5], float); pos += 5
        elif key in ("zone1_combos", "zone3_combos"):
            n = int(tok[pos]); pos += 1
            target = zone1_combo if key == "zone1_combos" else zone3_combo
            for _ in range(n):
                target[tok[pos]] = float(tok[pos + 1]); pos += 2
        elif key == "n_activity_sites":
            n_sites = int(tok[pos]); pos += 1
        elif key.startswith("zone") and key.endswith("_ptr"):
            if n_sites <= 0: raise ValueError(f"n_activity_sites missing before {key}")
            z = int(key[4])
            ptr[z] = np.asarray(tok[pos:pos + n_sites + 1], int); pos += n_sites + 1
        elif key.startswith("zone") and key.endswith("_indices"):
            z = int(key[4])
            n = int(tok[pos]); pos += 1
            indices[z] = np.asarray(tok[pos:pos + n], int); pos += n
        else:
            raise ValueError(f"Unknown model key {key} in {path}")
    if elements != list(ELEMENTS) or n_sites <= 0 or 1 not in ptr:
        raise ValueError(f"Incomplete model {path}")
    return ActivityModel(site_type, elements, intercept, e_opt, activity_temperature, zone1_element,
                         zone_coeff, zone1_combo, zone3_combo, ptr, indices, n_sites)


def atomic_numbers_to_occ(numbers) -> np.ndarray:
    try:
        return np.asarray([Z_TO_INDEX[int(z)] for z in numbers], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Unexpected atomic number {exc.args[0]}") from exc


def combo_key(occ: np.ndarray, sites: np.ndarray) -> str:
    return "".join(ELEMENTS[i] for i in sorted(int(occ[s]) for s in sites))


def evaluate_sites(model: ActivityModel, occ: np.ndarray, be_shifts: np.ndarray):
    final_be = np.empty(model.n_sites, float)
    linear_be = np.empty(model.n_sites, float)
    site_shift = np.empty(model.n_sites, float)
    fractions = np.zeros((model.n_sites, 5), float)
    zone1_sizes = set()
    for sidx in range(model.n_sites):
        z1 = model.indices[1][model.ptr[1][sidx]:model.ptr[1][sidx + 1]]
        z1_occ = occ[z1]
        zone1_sizes.add(len(z1))
        fractions[sidx] = np.bincount(z1_occ, minlength=5) / len(z1)
        e = model.intercept
        if model.site_type == "top":
            e += float(model.zone1_element[z1_occ].sum())
        else:
            key = combo_key(occ, z1)
            if key in model.zone1_combo:
                e += model.zone1_combo[key]
            else:
                e += float(model.zone1_element[z1_occ].sum())
        for zone in range(2, 6):
            if zone not in model.ptr:
                continue
            sites = model.indices[zone][model.ptr[zone][sidx]:model.ptr[zone][sidx + 1]]
            if zone == 3 and model.zone3_combo:
                key = combo_key(occ, sites)
                if key in model.zone3_combo:
                    e += model.zone3_combo[key]
                    continue
            if len(sites):
                e += float(model.zone_coeff[zone][occ[sites]].sum())
        shift = float(be_shifts[z1_occ].mean())
        linear_be[sidx] = e
        site_shift[sidx] = shift
        final_be[sidx] = e - shift
    return linear_be, site_shift, final_be, fractions, zone1_sizes


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def layer_groups(template_path: Path):
    atoms = read(template_path)
    keys = np.round(atoms.positions[:, 2], 6)
    groups = [np.flatnonzero(keys == value) for value in sorted(set(keys))]
    if len(groups) != 10 or any(len(group) != 100 for group in groups):
        raise ValueError(f"Unexpected layers in {template_path}: {[len(g) for g in groups]}")
    return groups


def shuffle_occ(occ: np.ndarray, groups, seed: int) -> np.ndarray:
    out = occ.copy()
    rng = np.random.default_rng(seed)
    for group in groups:
        values = out[group].copy()
        rng.shuffle(values)
        out[group] = values
    return out


def load_shift_data(case: str):
    root = SHIFT_ROOT / case / "kde_smoothed"
    source = root / "log_kde_representative_trial_BE_shifts.csv"
    source_row = pd.read_csv(source).iloc[0]
    rows = [{
        "element": element,
        "selected_be_shift_exact_eV": float(source_row[f"be_shift_{element}_exact_eV"]),
        "selected_trial": int(source_row["trial"]),
    } for element in ELEMENTS]
    be_df = pd.DataFrame(rows).set_index("element").loc[list(ELEMENTS)]
    be_shifts = be_df["selected_be_shift_exact_eV"].to_numpy(float)
    # These slabs remain equi-atomic. Composition-shift files are intentionally
    # neither loaded nor used in this analysis.
    composition = np.full(len(ELEMENTS), 1.0 / len(ELEMENTS))
    return be_df, be_shifts, composition


def reconstruct_random(workdir: Path, run: int, tmpdir: Path) -> np.ndarray:
    out = tmpdir / f"initial_{run:02d}.json"
    command = [str(RECONSTRUCT_EXE), "--ce-export", str(workdir / "inputs/ce_export.txt"),
               "--seeds", str(workdir / "results/random_slab_seeds.csv"),
               "--trial", "0", "--composition-index", "0", "--run", str(run),
               "--output", str(out)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return atomic_numbers_to_occ(json.loads(out.read_text()))


def load_cemc_records(workdir: Path, temperature: int) -> dict[int, np.ndarray]:
    path = workdir / "results/structures_by_temperature" / f"T{temperature:05d}" / "comp_0000.jsonl"
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["run"])] = atomic_numbers_to_occ(row["Z"])
    if sorted(records) != list(range(20)):
        raise ValueError(f"Missing runs in {path}: {sorted(records)}")
    return records


def smoothed_weighted_density(values, weights, edges, denominator):
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    width = edges[1] - edges[0]
    density = counts / (denominator * width)
    x = np.arange(-6, 7)
    kernel = np.exp(-0.5 * (x / 1.5) ** 2)
    kernel /= kernel.sum()
    return np.convolve(density, kernel, mode="same")


def weighted_stats(values, weights):
    mask = weights > 0
    v = values[mask]
    w = weights[mask]
    total = float(w.sum())
    if total <= 0.0:
        return 0, 0.0, math.nan, math.nan, math.nan
    mean = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    order = np.argsort(v)
    vv, ww = v[order], w[order]
    median = float(vv[np.searchsorted(np.cumsum(ww), total / 2)])
    return int(mask.sum()), total, mean, sd, median


def activity_no_log(delta_g_h: np.ndarray, activity_temperature: float) -> float:
    """C++ v1.0 activity: mean(exp(-abs(deltaG_H)/(k_B*T)))."""
    temperature = activity_temperature if activity_temperature > 0.0 else 298.0
    return float(np.mean(np.exp(-np.abs(delta_g_h) / (KB_EV_PER_K * temperature))))


def analyze_combo(facet: str, site: str, case: str, raw_output: bool, no_be_shift: bool = False):
    combo = f"{facet}/{site}"
    workdir = BASE / facet / site
    analysis_name = ("hbe_distribution_analysis_no_be_shift" if no_be_shift else ANALYSIS_1X3)
    analysis_dir = workdir / analysis_name
    plots_dir = analysis_dir / "plots_by_temperature"
    data_dir = analysis_dir / "site_data_by_temperature"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if raw_output: data_dir.mkdir(parents=True, exist_ok=True)
    for old_pdf in analysis_dir.glob("*.pdf"):
        old_pdf.unlink()
    model = parse_model(workdir / "inputs/activity_model.txt")
    groups = layer_groups(workdir / "inputs/template.cif")
    be_df, selected_be_shifts, composition = load_shift_data(case)
    be_shifts = np.zeros_like(selected_be_shifts) if no_be_shift else selected_be_shifts

    shift_rows = []
    for i, element in enumerate(ELEMENTS):
        shift_rows.append({
            "facet": facet, "site": site, "element": element,
            "selected_be_shift_exact_eV": selected_be_shifts[i],
            "applied_be_shift_eV": be_shifts[i],
            "selected_be_trial": int(be_df.iloc[i]["selected_trial"]),
            "composition_shift_applied": False,
            "composition_ratio_unshifted": composition[i],
        })
    pd.DataFrame(shift_rows).to_csv(analysis_dir / "selected_shifts_and_final_composition.csv", index=False)
    shift_source_dir = SHIFT_ROOT / case / "kde_smoothed"
    pd.read_csv(shift_source_dir / "log_kde_representative_trial_BE_shifts.csv").to_csv(
        analysis_dir / "selected_log_activity_KDE_peak_trial_BE_shifts.csv", index=False)
    pd.read_csv(shift_source_dir / "log_kde_representative_trial_composition_shifts.csv").to_csv(
        analysis_dir / "selected_log_activity_KDE_peak_trial_composition_shifts.csv", index=False)

    all_stats = []
    with tempfile.TemporaryDirectory(prefix=f"hbe_{facet}_{site}_") as tmp:
        tmpdir = Path(tmp)
        random_occ = [reconstruct_random(workdir, run, tmpdir) for run in range(20)]
        random_eval = [evaluate_sites(model, occ, be_shifts) for occ in random_occ]
        expected_zone1 = {1 if model.site_type == "top" else (2 if site == "bridge" else (4 if facet == "100" else 3))}
        found_sizes = set().union(*(row[4] for row in random_eval))
        if found_sizes != expected_zone1:
            raise ValueError(f"{combo}: expected zone1 size {expected_zone1}, found {found_sizes}")

        all_temperature_values = {method: [] for method in METHODS}
        all_temperature_fractions = {method: [] for method in METHODS}
        with NoPdfOutput() as pdf:
            for temperature in TEMPERATURES:
                cemc = load_cemc_records(workdir, temperature)
                eval_by_method = {"Random": random_eval, "CEMC": [], "CEMC + layer shuffle": []}
                for run in range(20):
                    occ = cemc[run]
                    eval_by_method["CEMC"].append(evaluate_sites(model, occ, be_shifts))
                    seed = stable_seed("FeCoNiPdPt-equi-layer-shuffle-v1", combo, run, temperature)
                    shuffled = shuffle_occ(occ, groups, seed)
                    eval_by_method["CEMC + layer shuffle"].append(evaluate_sites(model, shuffled, be_shifts))

                packed = {}
                all_values = []
                for method in METHODS:
                    linear = np.concatenate([r[0] for r in eval_by_method[method]])
                    shifts = np.concatenate([r[1] for r in eval_by_method[method]])
                    final_be = np.concatenate([r[2] for r in eval_by_method[method]])
                    delta_g_h = final_be - model.e_opt
                    fractions = np.vstack([r[3] for r in eval_by_method[method]])
                    packed[method] = (linear, shifts, delta_g_h, fractions)
                    all_values.append(delta_g_h)
                    all_temperature_values[method].append(delta_g_h)
                    all_temperature_fractions[method].append(fractions)
                activities = {
                    method: activity_no_log(packed[method][2], model.activity_temperature)
                    for method in METHODS
                }
                combined = np.concatenate(all_values)
                lo, hi = float(combined.min()), float(combined.max())
                lo = min(lo, OPTIMAL_DELTA_G_H_EV) - 0.06
                hi = max(hi, OPTIMAL_DELTA_G_H_EV) + 0.06
                edges = np.linspace(lo, hi, 91)
                centers = (edges[:-1] + edges[1:]) / 2

                n_panels = 5 if no_be_shift else 3
                fig, axes = plt.subplots(1, n_panels, figsize=((25, 6.2) if no_be_shift else (18.5, 6.4)),
                                         sharex=True, sharey=True)
                for method in METHODS:
                    linear, shifts, values, fractions = packed[method]
                    for ei, element in enumerate(ELEMENTS):
                        weights = fractions[:, ei]
                        density = smoothed_weighted_density(values, weights, edges, len(values))
                        if no_be_shift:
                            ax = axes[ei]
                            ax.plot(centers, density, lw=2.6, color=COLORS[method], label=method)
                            ax.fill_between(centers, density, color=COLORS[method], alpha=0.10)
                        else:
                            ax = axes[METHODS.index(method)]
                            ax.plot(centers, density, lw=2.6, color=ELEMENT_COLORS[element], label=element)
                        n_ensembles, total_weight, mean, sd, median = weighted_stats(values, weights)
                        all_stats.append({
                            "facet": facet, "site": site, "temperature_K": temperature,
                            "method": method, "element": element,
                            "n_all_adsorption_sites": len(values),
                            "n_ensembles_containing_element": n_ensembles,
                            "sum_fractional_zone1_weight": total_weight,
                            "integrated_fractional_contribution": total_weight / len(values),
                            "weighted_mean_deltaG_H_eV": mean,
                            "weighted_sd_deltaG_H_eV": sd,
                            "weighted_median_deltaG_H_eV": median,
                            "selected_BE_shift_eV": be_shifts[ei],
                            "composition_shift_applied": False,
                            "composition_ratio_unshifted": composition[ei],
                            "optimal_deltaG_H_eV": OPTIMAL_DELTA_G_H_EV,
                            "activity_no_log": activities[method],
                            "activity_temperature_K": model.activity_temperature,
                        })
                if no_be_shift:
                    for ei, (element, ax) in enumerate(zip(ELEMENTS, axes)):
                        ax.axvline(OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.0, color="black",
                                   label=r"Optimal $\Delta G_{\mathrm{H}}$" if ei == 0 else None)
                        ax.set_title(element, pad=9)
                        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
                        ax.tick_params(width=1.2, length=5)
                        ax.grid(alpha=0.20)
                else:
                    for method, ax in zip(METHODS, axes):
                        ax.axvline(OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.0, color="black",
                                   label=r"Optimal $\Delta G_{\mathrm{H}}$")
                        ax.set_title(f"{method}\nActivity = {activities[method]:.6f}", pad=9)
                        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
                        ax.tick_params(width=1.2, length=5)
                        ax.grid(alpha=0.20)
                axes[0].set_ylabel("Fraction-weighted ensemble density (eV$^{-1}$)")
                temp_label = "final 298 K" if temperature == 298 else f"{temperature} K"
                fig.suptitle(f"FeCoNiPdPt fcc({facet}) {site}: {temp_label}\n"
                             fr"$\Delta G_{{\mathrm{{H}}}}$ = H BE - ({model.e_opt:+.3f} eV)"
                             "\nActivity: mean over 20 slabs; no natural log", fontsize=18, y=1.08)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
                           ncol=(4 if no_be_shift else 6), frameon=False)
                fig.tight_layout(rect=(0, 0.09, 1, 0.96))
                stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
                fig.savefig(plots_dir / f"{stem}_element_HBE_distribution.png", dpi=180, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                if raw_output:
                    out_path = data_dir / f"{stem}_site_HBE.csv.gz"
                    with gzip.open(out_path, "wt", newline="") as handle:
                        fieldnames = ["facet", "site", "temperature_K", "method", "run", "site_index",
                                      "linear_H_BE_eV", "site_BE_shift_eV", "final_H_BE_eV"] + [f"zone1_fraction_{e}" for e in ELEMENTS]
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        for method in METHODS:
                            for run, result in enumerate(eval_by_method[method]):
                                linear, shifts, values, fractions = result[:4]
                                for sidx in range(model.n_sites):
                                    row = {"facet": facet, "site": site, "temperature_K": temperature,
                                           "method": method, "run": run, "site_index": sidx,
                                           "linear_H_BE_eV": linear[sidx], "site_BE_shift_eV": shifts[sidx],
                                           "final_H_BE_eV": values[sidx]}
                                    row.update({f"zone1_fraction_{e}": fractions[sidx, i] for i, e in enumerate(ELEMENTS)})
                                    writer.writerow(row)

        packed_all = {
            method: (np.concatenate(all_temperature_values[method]),
                     np.vstack(all_temperature_fractions[method]))
            for method in METHODS
        }
        activities_all = {
            method: activity_no_log(packed_all[method][0], model.activity_temperature)
            for method in METHODS
        }
        combined_all = np.concatenate([packed_all[method][0] for method in METHODS])
        lo = min(float(combined_all.min()), OPTIMAL_DELTA_G_H_EV) - 0.06
        hi = max(float(combined_all.max()), OPTIMAL_DELTA_G_H_EV) + 0.06
        edges = np.linspace(lo, hi, 91)
        centers = (edges[:-1] + edges[1:]) / 2
        averaged_rows = []
        n_panels = 5 if no_be_shift else 3
        fig, axes = plt.subplots(1, n_panels, figsize=((25, 6.2) if no_be_shift else (18.5, 6.4)),
                                 sharex=True, sharey=True)
        for method in METHODS:
            values, fractions = packed_all[method]
            for ei, element in enumerate(ELEMENTS):
                weights = fractions[:, ei]
                density = smoothed_weighted_density(values, weights, edges, len(values))
                if no_be_shift:
                    ax = axes[ei]
                    ax.plot(centers, density, lw=2.6, color=COLORS[method], label=method)
                    ax.fill_between(centers, density, color=COLORS[method], alpha=0.10)
                else:
                    ax = axes[METHODS.index(method)]
                    ax.plot(centers, density, lw=2.6, color=ELEMENT_COLORS[element], label=element)
                n_ensembles, total_weight, mean, sd, median = weighted_stats(values, weights)
                averaged_rows.append({
                    "facet": facet, "site": site, "temperature_scope": "all_19_temperatures_equal_weight",
                    "method": method, "element": element,
                    "n_temperatures": len(TEMPERATURES),
                    "n_all_adsorption_sites": len(values),
                    "n_ensembles_containing_element": n_ensembles,
                    "sum_fractional_zone1_weight": total_weight,
                    "integrated_fractional_contribution": total_weight / len(values),
                    "weighted_mean_deltaG_H_eV": mean,
                    "weighted_sd_deltaG_H_eV": sd,
                    "weighted_median_deltaG_H_eV": median,
                    "activity_no_log": activities_all[method],
                    "activity_temperature_K": model.activity_temperature,
                })
        if no_be_shift:
            for ei, (element, ax) in enumerate(zip(ELEMENTS, axes)):
                ax.axvline(OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.0, color="black",
                           label=r"Optimal $\Delta G_{\mathrm{H}}$" if ei == 0 else None)
                ax.set_title(element, pad=9)
                ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
                ax.tick_params(width=1.2, length=5)
                ax.grid(alpha=0.20)
        else:
            for method, ax in zip(METHODS, axes):
                ax.axvline(OPTIMAL_DELTA_G_H_EV, ls="--", lw=2.0, color="black",
                           label=r"Optimal $\Delta G_{\mathrm{H}}$")
                ax.set_title(f"{method}\nActivity = {activities_all[method]:.6f}", pad=9)
                ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
                ax.tick_params(width=1.2, length=5)
                ax.grid(alpha=0.20)
        axes[0].set_ylabel("Fraction-weighted ensemble density (eV$^{-1}$)")
        fig.suptitle(f"FeCoNiPdPt fcc({facet}) {site}: average over all 19 temperatures\n"
                     fr"$\Delta G_{{\mathrm{{H}}}}$ = H BE - ({model.e_opt:+.3f} eV)"
                     "\nActivity: mean over 19 temperatures x 20 slabs; no natural log",
                     fontsize=18, y=1.08)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
                   ncol=(4 if no_be_shift else 6), frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 0.96))
        fig.savefig(plots_dir / "all_temperatures_average_element_deltaG_H_distribution.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(averaged_rows).to_csv(
            analysis_dir / "all_temperatures_average_element_deltaG_H_summary.csv", index=False)

    stats = pd.DataFrame(all_stats)
    stats.to_csv(analysis_dir / "element_HBE_distribution_summary.csv", index=False)
    metadata = {
        "facet": facet, "site": site, "case": case, "n_runs": 20,
        "n_activity_sites_per_slab": model.n_sites,
        "temperatures_K": list(TEMPERATURES), "methods": list(METHODS),
        "reference_optimal_H_BE_eV": model.e_opt,
        "deltaG_H_formula": "final_H_BE_eV - reference_optimal_H_BE_eV",
        "optimal_deltaG_H_eV": OPTIMAL_DELTA_G_H_EV,
        "activity_formula_no_log": "mean(exp(-abs(deltaG_H)/(k_B*T_activity)))",
        "activity_temperature_K": model.activity_temperature,
        "all_temperatures_average_rule": "equal weight over all adsorption sites at each of 19 temperatures",
        "BE_shift_applied": not no_be_shift,
        "BE_shift_source": str(shift_source_dir / "log_kde_representative_trial_BE_shifts.csv"),
        "composition_shift_source": str(shift_source_dir / "log_kde_representative_trial_composition_shifts.csv"),
        "representative_selection": "maximum 3D-histogram KDE probability for CEMC log-activity with tau > 0",
        "selected_BE_shifts_eV": dict(zip(ELEMENTS, selected_be_shifts)),
        "applied_BE_shifts_eV": dict(zip(ELEMENTS, be_shifts)),
        "composition_shift_applied": False,
        "composition_unshifted": dict(zip(ELEMENTS, composition)),
        "composition_sum": float(composition.sum()),
        "non_top_element_weighting": "zone1 fractional contribution; e.g. FePt bridge = 0.5 Fe + 0.5 Pt",
        "site_BE_formula": ("linear_model_H_BE" if no_be_shift else
                            "linear_model_H_BE - mean(selected_BE_shift of zone1 atoms)"),
        "distribution_normalization": "weighted count / (all adsorption sites × bin width); curve area is zone1 elemental fraction",
    }
    (analysis_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return stats, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-raw-site-data", action="store_true")
    parser.add_argument("--no-be-shift", action="store_true")
    parser.add_argument("--only-case", choices=[case for _, _, case in COMBOS])
    args = parser.parse_args()
    all_stats = []
    all_metadata = []
    for facet, site, case in COMBOS:
        analysis_name = ("hbe_distribution_analysis_no_be_shift" if args.no_be_shift else ANALYSIS_1X3)
        analysis_dir = BASE / facet / site / analysis_name
        if args.only_case and case != args.only_case:
            stats = pd.read_csv(analysis_dir / "element_HBE_distribution_summary.csv")
            metadata = json.loads((analysis_dir / "analysis_metadata.json").read_text())
        else:
            stats, metadata = analyze_combo(facet, site, case, not args.no_raw_site_data, args.no_be_shift)
        all_stats.append(stats)
        all_metadata.append(metadata)
        print(f"Ready {case}: {len(stats)} summary rows")
    combined = pd.concat(all_stats, ignore_index=True)
    combined_name = ("H_BE_distribution_summary_all_facets_sites_no_be_shift.csv"
                     if args.no_be_shift else
                     "H_BE_distribution_summary_log_activity_KDE_peak_trial_BE_shift_1x3.csv")
    combined.to_csv(BASE / combined_name, index=False)
    overall = {
        "combinations": [case for _, _, case in COMBOS],
        "n_figures_png": len(COMBOS) * (len(TEMPERATURES) + 1),
        "n_multipage_pdf": 0,
        "BE_shift_applied": not args.no_be_shift,
        "n_site_data_files_gz": sum(1 for f, s, _ in COMBOS for _ in (BASE / f / s / analysis_name / "site_data_by_temperature").glob("*.csv.gz")),
        "n_summary_rows": len(combined),
        "validated": True,
        "per_combination": all_metadata,
    }
    overall_name = ("H_BE_distribution_analysis_all_no_be_shift.json"
                    if args.no_be_shift else
                    "H_BE_distribution_analysis_log_activity_KDE_peak_trial_BE_shift_1x3.json")
    complete_name = ("H_BE_DISTRIBUTION_ANALYSIS_NO_BE_SHIFT_COMPLETE"
                     if args.no_be_shift else
                     "H_BE_DISTRIBUTION_LOG_ACTIVITY_KDE_PEAK_TRIAL_BE_SHIFT_1X3_COMPLETE")
    (BASE / overall_name).write_text(json.dumps(overall, indent=2) + "\n")
    (BASE / complete_name).touch()
    print(json.dumps({k: v for k, v in overall.items() if k != "per_combination"}, indent=2))


if __name__ == "__main__":
    main()
