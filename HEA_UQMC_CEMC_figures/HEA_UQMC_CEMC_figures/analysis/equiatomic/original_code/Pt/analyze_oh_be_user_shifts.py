#!/usr/bin/env python3
"""OH BE distributions and paired slab bootstrap for the requested BE shifts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np


BASE = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_oh_be_distributions as common  # noqa: E402

SLABS = BASE / "slabs"
OUT = BASE / "oh_be_distribution_user_shift_20260824"
ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")
TEMPERATURES = tuple(range(2000, 299, -100)) + (298,)
SHIFTS = {"Ir": 0.44736, "Pd": 0.36324, "Pt": -0.00136,
          "Rh": -0.05953, "Ru": 0.28397}
OPTIMUM = 1.1
WINDOW = (1.0, 1.2)
N_RUNS = 20
N_BOOT = 10_000
COLORS = common.ELEMENT_COLORS


def occupancy(path: Path) -> np.ndarray:
    # These files are P1 CIFs written in CE index order.  Reading the first
    # atom-site column directly avoids ASE's costly generic CIF symmetry pass
    # while preserving exactly the same 1000-element occupancy vector.
    values = []
    in_atoms = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "_atom_site_occupancy":
            in_atoms = True
            continue
        if in_atoms:
            fields = stripped.split()
            if fields and fields[0] in ELEMENTS and len(fields) >= 7:
                values.append(fields[0])
            elif values and stripped and not stripped.startswith("#"):
                break
    if len(values) != 1000:
        raise ValueError(f"{path}: expected 1000 atom rows, found {len(values)}")
    return np.asarray(values, dtype="U2")


def cemc_path(run: int, temperature: int) -> Path:
    folder = SLABS / f"slab_{run + 1:02d}"
    name = "final_0298K_cemc.cif" if temperature == 298 else f"T_{temperature:04d}K_cemc.cif"
    return folder / name


def smooth_density(values, labels, element, edges):
    return common.smooth_density(values, labels, element, edges)[0]


def probability_mass(values, labels, element, edges):
    selected = values[labels == element]
    counts, _ = np.histogram(selected, bins=edges)
    return counts.astype(float) / counts.sum()


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.titlesize": 21, "axes.labelsize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 15, "axes.linewidth": 1.2,
    })


def plot_1x2(random_pack, cemc_pack, edges, activity_temperature, output):
    set_style()
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4), sharex=True, sharey=True)
    for ax, title, (values, labels) in zip(axes, ("Homogeneous", "CEMC"), (random_pack, cemc_pack)):
        for element in ELEMENTS:
            ax.plot(centers, smooth_density(values, labels, element, edges),
                    color=COLORS[element], lw=2.7, label=element)
        ax.axvline(OPTIMUM, color="black", lw=2.1, ls="--")
        ax.set_title(title, pad=10)
        ax.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
        ax.grid(False)
        ax.tick_params(direction="out", width=1.2, length=5)
        activity = common.activity_no_log(values, OPTIMUM, activity_temperature)
        activity_x, activity_y = ((0.38, 0.68) if title == "Homogeneous" else (0.04, 0.95))
        ax.text(
            activity_x, activity_y, f"Activity = {activity:.6f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=17,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3.0),
        )
    axes[0].set_ylabel("Probability density")
    handles = [Line2D([0], [0], color=COLORS[e], lw=2.7, label=e) for e in ELEMENTS]
    legend = axes[0].legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.98),
        ncol=2, frameon=True, framealpha=0.92, facecolor="white", edgecolor="none",
        columnspacing=1.4, handlelength=2.4, fontsize=18,
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_1x5(random_pack, cemc_pack, edges, output):
    set_style()
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 32,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.linewidth": 0.8,
    })
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    fig, axes = plt.subplots(1, 5, figsize=(24.5, 5.8), sharex=True, sharey=True)
    containment = {}
    rows = []
    for ax, element in zip(axes, ELEMENTS):
        r_mass = probability_mass(*random_pack, element, edges)
        c_mass = probability_mass(*cemc_pack, element, edges)
        support = r_mass > 0
        inside = np.where(support, c_mass, 0.0)
        outside = np.where(~support, c_mass, 0.0)
        containment[element] = float(inside.sum())
        ax.axvspan(*WINDOW, color="#808080", alpha=0.15, zorder=0)
        ax.bar(centers, inside, width=width * 0.92, color="#4C78A8", alpha=0.80,
               linewidth=0, zorder=1)
        ax.bar(centers, outside, width=width * 0.92, color="#E45756", alpha=0.82,
               linewidth=0, zorder=1)
        ax.stairs(r_mass, edges, color="black", lw=2.0, zorder=3)
        ax.axvline(OPTIMUM, color="black", ls="--", lw=1.9, zorder=4)
        ax.set_title(element, pad=8)
        ax.set_xlabel(r"$\Delta E_{\mathrm{OH}}$ (eV)")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.18)
        ax.text(0.04, 0.94, f"$C_{{{element}}}$ = {100 * containment[element]:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=28,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.5))
        for i in range(len(centers)):
            rows.append({"element": element, "bin_left_eV": edges[i],
                         "bin_right_eV": edges[i + 1], "bin_center_eV": centers[i],
                         "random_probability_mass": r_mass[i],
                         "cemc_probability_mass": c_mass[i],
                         "random_support": int(support[i]),
                         "cemc_inside_random_support_mass": inside[i],
                         "cemc_outside_random_support_mass": outside[i],
                         "in_optimal_window_1p0_to_1p2_eV": int(
                             edges[i + 1] > WINDOW[0] and edges[i] < WINDOW[1])})
    axes[0].set_ylabel("Probability mass")
    legend = [Line2D([0], [0], color="black", lw=2, label="Random"),
              Patch(facecolor="#4C78A8", alpha=.8, label="CEMC inside Random distribution"),
              Patch(facecolor="#E45756", alpha=.82, label="CEMC outside Random distribution"),
              Patch(facecolor="#808080", alpha=.15, label="Optimal window"),
              Line2D([0], [0], color="black", lw=1.9, ls="--", label="Optimal")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, .82), w_pad=1.1)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return containment, rows


def bootstrap(random_by_run, cemc_by_run, edges):
    rng = np.random.default_rng(common.stable_seed(
        "PtPdRhRuIr-OH-user-shift-paired-joint-bootstrap-v1", *SHIFTS.values()))
    sampled = rng.integers(0, N_RUNS, size=(N_BOOT, N_RUNS))
    result = np.empty((N_BOOT, len(ELEMENTS)))
    full = np.empty(len(ELEMENTS))
    for j, element in enumerate(ELEMENTS):
        r_presence = np.zeros((N_RUNS, len(edges) - 1), bool)
        c_counts = np.zeros((N_RUNS, len(edges) - 1), float)
        for run in range(N_RUNS):
            rv, rl = random_by_run[run]
            cv, cl = cemc_by_run[run]
            r_presence[run] = np.histogram(rv[rl == element], bins=edges)[0] > 0
            c_counts[run] = np.histogram(cv[cl == element], bins=edges)[0]
        support = r_presence.any(axis=0)
        full[j] = c_counts[:, support].sum() / c_counts.sum()
        for b in range(N_BOOT):
            idx = sampled[b]
            bs_support = r_presence[idx].any(axis=0)
            bs_counts = c_counts[idx].sum(axis=0)
            result[b, j] = bs_counts[bs_support].sum() / bs_counts.sum()
    return full, result


def plot_bootstrap(full, samples, output):
    set_style()
    mean_samples = samples.mean(axis=1)
    labels = list(ELEMENTS) + ["Element mean"]
    series = [samples[:, i] for i in range(len(ELEMENTS))] + [mean_samples]
    point = list(full) + [float(full.mean())]
    colors = [COLORS[e] for e in ELEMENTS] + ["#333333"]
    lows = [np.percentile(x, 2.5) for x in series]
    highs = [np.percentile(x, 97.5) for x in series]
    xmin = max(0.0, min(np.percentile(x, .05) for x in series) - .025)
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.axvspan(.90, 1.0, color="#55A868", alpha=.12, label="≥90% containment")
    rng = np.random.default_rng(20260824)
    for i, (label, x, c, p, lo, hi) in enumerate(zip(labels, series, colors, point, lows, highs)):
        y = len(labels) - 1 - i
        violin = ax.violinplot(x, positions=[y], widths=.70, showextrema=False,
                               quantiles=[[.025, .5, .975]], vert=False)
        for body in violin["bodies"]:
            body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(.22)
        if "cquantiles" in violin:
            violin["cquantiles"].set_color(c); violin["cquantiles"].set_linewidth(1.5)
        ax.hlines(y, lo, hi, color=c, lw=3.0, zorder=4)
        ax.plot([lo, hi], [y, y], "|", color=c, ms=12, mew=2.2, zorder=5)
        marker = "D" if label == "Element mean" else "o"
        ax.scatter(p, y, s=105, marker=marker, color=c, edgecolor="white", linewidth=1.0, zorder=6)
        ax.plot(np.mean(x), y, marker="|", color="black", ms=18, mew=2.0, zorder=7)
        ax.text(min(1.015, hi + .008), y, f"{100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]",
                va="center", fontsize=14)
    ax.axvline(.90, color="#2F6B3B", ls="--", lw=1.8)
    ax.set_xlim(xmin, 1.035)
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.set_xlabel("CEMC probability mass contained in Random occupied-bin support")
    fig.suptitle("Paired joint slab bootstrap: Random vs CEMC", fontsize=24, y=.985)
    fig.text(.5, .945,
             "10,000 paired run-cluster resamples; all 19 CEMC temperatures retained per sampled slab",
             ha="center", va="center", fontsize=14)
    ax.grid(axis="x", alpha=.20)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
                      markeredgecolor="white", markersize=9, label="Full-data estimate"),
               Line2D([0], [0], marker="|", color="black", markersize=14,
                      linestyle="none", label="Bootstrap mean"),
               Line2D([0], [0], color="#555555", lw=3, label="95% percentile CI"),
               Patch(facecolor="#55A868", alpha=.12, label="≥90% containment")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .012),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .105, 1, .925))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rows = []
    for label, x, p in zip(labels, series, point):
        rows.append({"element": label, "full_data_containment": p,
                     "bootstrap_mean": float(np.mean(x)),
                     "bootstrap_sd": float(np.std(x, ddof=1)),
                     "ci_2p5": float(np.percentile(x, 2.5)),
                     "ci_97p5": float(np.percentile(x, 97.5)),
                     "n_bootstrap": N_BOOT})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = common.parse_model(common.MODEL_PATH)
    surface, _ = common.surface_sites_and_validate(model)
    if abs(model["e_opt"] - OPTIMUM) > 1e-12:
        raise ValueError(f"model optimum {model['e_opt']} != {OPTIMUM}")
    random_by_run, cemc_by_run = [], []
    source_rows = []
    for run in range(N_RUNS):
        rp = SLABS / f"slab_{run + 1:02d}" / "initial_random.cif"
        if not rp.exists(): raise FileNotFoundError(rp)
        random_by_run.append(common.evaluate(model, surface, occupancy(rp), SHIFTS))
        vals, labs = [], []
        for t in TEMPERATURES:
            cp = cemc_path(run, t)
            if not cp.exists(): raise FileNotFoundError(cp)
            v, l = common.evaluate(model, surface, occupancy(cp), SHIFTS)
            vals.append(v); labs.append(l)
            source_rows.append({"run": run, "temperature_K": t, "random_cif": str(rp),
                                "cemc_cif": str(cp), "n_OH_sites": len(v)})
        cemc_by_run.append((np.concatenate(vals), np.concatenate(labs)))
    random_pack = (np.concatenate([x[0] for x in random_by_run]),
                   np.concatenate([x[1] for x in random_by_run]))
    cemc_pack = (np.concatenate([x[0] for x in cemc_by_run]),
                 np.concatenate([x[1] for x in cemc_by_run]))
    all_values = np.concatenate((random_pack[0], cemc_pack[0], [OPTIMUM, *WINDOW]))
    edges = np.linspace(float(all_values.min()) - .06, float(all_values.max()) + .06, 91)
    plot_1x2(random_pack, cemc_pack, edges, model["activity_temperature"],
             OUT / "oh_be_distribution_1x2_random_cemc.png")
    containment, bin_rows = plot_1x5(random_pack, cemc_pack, edges,
                                     OUT / "probability_mass_weighted_histograms_1x5.png")
    full, samples = bootstrap(random_by_run, cemc_by_run, edges)
    boot_rows = plot_bootstrap(full, samples, OUT / "paired_joint_bootstrap.png")
    write_csv(OUT / "histogram_bin_data.csv", bin_rows)
    write_csv(OUT / "bootstrap_summary.csv", boot_rows)
    write_csv(OUT / "input_structure_manifest.csv", source_rows)
    np.savez_compressed(OUT / "bootstrap_replicates.npz", elements=np.array(ELEMENTS),
                        containment=samples, element_mean=samples.mean(axis=1),
                        full_data=full, bin_edges_eV=edges)
    stats = []
    for method, pack in (("Random", random_pack), ("CEMC", cemc_pack)):
        for e in ELEMENTS:
            x = pack[0][pack[1] == e]
            stats.append({"method": method, "element": e, "n_sites": len(x),
                          "mean_OH_BE_eV": float(x.mean()), "sd_OH_BE_eV": float(x.std(ddof=1)),
                          "median_OH_BE_eV": float(np.median(x))})
    write_csv(OUT / "oh_be_summary_statistics.csv", stats)
    metadata = {
        "formula": "E_OH = intercept + zone1(top) + sum(zone2) + sum(zone3) - BE_shift(top element)",
        "BE_shifts_eV": SHIFTS, "optimal_OH_binding_energy_eV": OPTIMUM,
        "optimal_window_eV": list(WINDOW), "elements": list(ELEMENTS),
        "temperatures_K": list(TEMPERATURES), "n_random_slabs": N_RUNS,
        "n_cemc_structures": N_RUNS * len(TEMPERATURES), "OH_sites_per_structure": 100,
        "histogram_bins": len(edges) - 1, "common_bin_edges_eV": edges.tolist(),
        "containment_full_data": containment,
        "bootstrap": {"n_replicates": N_BOOT, "unit": "slab run",
                      "paired_random_cemc": True,
                      "cemc_temperatures_retained_per_run": len(TEMPERATURES)},
        "input_root": str(SLABS), "model": str(common.MODEL_PATH),
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validation = {"status": "PASS", "random_site_count": len(random_pack[0]),
                  "cemc_site_count": len(cemc_pack[0]),
                  "all_finite": bool(np.isfinite(np.concatenate((random_pack[0], cemc_pack[0]))).all()),
                  "random_expected": 20 * 100, "cemc_expected": 20 * 19 * 100,
                  "bootstrap_shape": list(samples.shape),
                  "mass_sums_one": all(abs(sum(r["random_probability_mass"] for r in bin_rows if r["element"] == e)-1)<1e-12 and abs(sum(r["cemc_probability_mass"] for r in bin_rows if r["element"] == e)-1)<1e-12 for e in ELEMENTS)}
    if validation["random_site_count"] != validation["random_expected"] or validation["cemc_site_count"] != validation["cemc_expected"] or not validation["all_finite"] or not validation["mass_sums_one"]:
        validation["status"] = "FAIL"
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "validation": validation, "bootstrap": boot_rows}, indent=2))


if __name__ == "__main__":
    main()
