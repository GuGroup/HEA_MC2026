#!/usr/bin/env python3
"""PtPdIrRhRu OH-BE distributions and paired-joint slab bootstrap for new shifts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import analyze_oh_be_distributions as base


FIT_ROOT = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/fit_exp")
RESULT_ROOT = base.BASE / "log_activity_kde_representative_trial_all_ML"
BE_TABLE = FIT_ROOT / "log_activity_kde_representative_trial_BE_shifts_all_ML.csv"
COMP_TABLE = FIT_ROOT / "log_activity_kde_representative_trial_composition_shifts_all_ML.csv"
ML_NAME = ""
SHIFTS = {}
OUT_1X3 = OUT_1X5 = OUT_BOOT = PLOTS_1X3 = PLOTS_1X5 = None
N_RUNS = 20
N_BOOTSTRAP = 10_000
N_BINS = 90
METHOD_COLORS = {
    "Random": "#555555", "CEMC": "#2468B4", "CEMC + layer shuffle": "#E68613"
}


def configure(ml_name):
    global ML_NAME, SHIFTS, OUT_1X3, OUT_1X5, OUT_BOOT, PLOTS_1X3, PLOTS_1X5
    ML_NAME = ml_name
    frame = pd.read_csv(BE_TABLE)
    row = frame.loc[frame["case"] == ml_name].iloc[0]
    SHIFTS = {element: float(row[f"be_shift_{element}_exact_eV"])
              for element in base.ELEMENTS}
    root = RESULT_ROOT / ml_name
    OUT_1X3 = root / "OH_BE_distribution_1x3"
    OUT_1X5 = root / "OH_BE_probability_mass_1x5_random_support"
    OUT_BOOT = root / "paired_joint_slab_bootstrap"
    PLOTS_1X3 = OUT_1X3 / "plots_by_temperature"
    PLOTS_1X5 = OUT_1X5 / "plots_by_temperature"
    root.mkdir(parents=True, exist_ok=True)
    frame.loc[frame["case"] == ml_name].to_csv(
        root / "selected_log_activity_KDE_peak_trial_BE_shifts.csv", index=False)
    comp = pd.read_csv(COMP_TABLE)
    comp.loc[comp["case"] == ml_name].to_csv(
        root / "selected_log_activity_KDE_peak_trial_composition_shifts.csv", index=False)
    return row


def packed(evaluated):
    return {
        method: (np.concatenate([row[0] for row in evaluated[method]]),
                 np.concatenate([row[1] for row in evaluated[method]]))
        for method in base.METHODS
    }


def activities(evaluated, model):
    return {
        method: np.asarray([
            base.activity_no_log(row[0], model["e_opt"], model["activity_temperature"])
            for row in evaluated[method]
        ])
        for method in base.METHODS
    }


def common_edges(data, optimum, n_bins=N_BINS, methods=base.METHODS):
    values = np.concatenate([data[m][0] for m in methods])
    lo = min(float(values.min()), optimum) - 0.08
    hi = max(float(values.max()), optimum) + 0.08
    return np.linspace(lo, hi, n_bins + 1)


def plot_1x3(data, activity, title, destination, model):
    edges = common_edges(data, model["e_opt"], 100)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    })
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.8), sharex=True, sharey=True)
    for method, ax in zip(base.METHODS, axes):
        values, labels = data[method]
        for element in base.ELEMENTS:
            density, _ = base.smooth_density(values, labels, element, edges)
            ax.plot(centers, density, lw=2.7, color=base.ELEMENT_COLORS[element], label=element)
        ax.axvline(model["e_opt"], color="black", ls="--", lw=2.2,
                   label="Optimal OH BE = 1.1 eV")
        ax.set_title(
            f"{method}\nActivity = {np.mean(activity[method]):.6f} ± {np.std(activity[method]):.6f}",
            pad=10,
        )
        ax.set_xlabel("OH BE (eV)")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("Surface-fraction-weighted site density (eV$^{-1}$)")
    fig.suptitle(title, fontsize=19, y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025),
               ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def probability_mass(values, labels, element, edges):
    selected = values[labels == element]
    counts, _ = np.histogram(selected, bins=edges)
    mass = (counts.astype(float) / float(len(selected))
            if len(selected) else np.zeros(len(edges) - 1, dtype=float))
    return mass, len(selected)


def plot_1x5_mass(data, activity, title, destination, model, save_bins=False):
    """Plot CEMC mass inside/outside Random occupied-bin support."""
    del activity  # Activity is shown in the separate 1x3 method comparison.
    edges = common_edges(data, model["e_opt"], methods=("Random", "CEMC"))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    rows = []
    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 19, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    })
    fig, axes = plt.subplots(1, 5, figsize=(27.5, 7.3), sharex=True, sharey=True)
    for element, ax in zip(base.ELEMENTS, axes):
        random_mass, n_random = probability_mass(
            data["Random"][0], data["Random"][1], element, edges)
        cemc_mass, n_cemc = probability_mass(
            data["CEMC"][0], data["CEMC"][1], element, edges)
        random_support = random_mass > 0
        cemc_inside = np.where(random_support, cemc_mass, 0.0)
        cemc_outside = np.where(random_support, 0.0, cemc_mass)
        containment = float(np.sum(cemc_inside))

        for b in np.flatnonzero(random_support):
            ax.axvspan(edges[b], edges[b + 1], color="#e8e8e8", alpha=0.72,
                       linewidth=0, zorder=0)
        ax.bar(centers, cemc_inside, width=0.88 * widths, color="#4f88c6",
               alpha=0.88, edgecolor="none", zorder=2)
        ax.bar(centers, cemc_outside, width=0.88 * widths, color="#c85e64",
               alpha=0.88, edgecolor="none", zorder=2)
        ax.step(edges[:-1], random_mass, where="post", lw=2.15,
                color="#303030", zorder=3)
        ax.plot([edges[-1], edges[-1]], [random_mass[-1], 0],
                color="#303030", lw=2.15, zorder=3)
        ax.text(0.98, 0.95,
                rf"$C_{{\mathrm{{{element}}}}} = {100 * containment:.2f}\%$",
                transform=ax.transAxes, fontsize=17, ha="right", va="top")
        if n_random == 0 or n_cemc == 0:
            missing = []
            if n_random == 0:
                missing.append("Random")
            if n_cemc == 0:
                missing.append("CEMC")
            ax.text(0.03, 0.87, f"No {element} top sites: {', '.join(missing)}",
                    transform=ax.transAxes, fontsize=11, ha="left", va="top")
        if save_bins:
            for b in range(len(centers)):
                rows.append({
                    "element": element, "bin_index": b,
                    "bin_left_eV": edges[b], "bin_right_eV": edges[b + 1],
                    "bin_center_eV": centers[b],
                    "random_probability_mass": random_mass[b],
                    "random_occupied_bin_support": bool(random_support[b]),
                    "cemc_probability_mass": cemc_mass[b],
                    "cemc_mass_inside_random_support": cemc_inside[b],
                    "cemc_mass_outside_random_support": cemc_outside[b],
                    "containment_percent": 100 * containment,
                    "n_random_element_OH_sites": n_random,
                    "n_cemc_element_OH_sites": n_cemc,
                })
        ax.axvline(model["e_opt"], color="black", ls="--", lw=2.2)
        ax.set_title(element, pad=9)
        ax.set_xlabel("OH BE (eV)")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("Fractional probability mass per bin")
    formula = (r"$p_{\mathrm{CEMC},b,e}=\dfrac{\sum_{i\in b}w_{i,e}}"
               r"{\sum_i w_{i,e}};\quad C_e=\sum_b I_{\mathrm{RS},b,e}"
               r"p_{\mathrm{CEMC},b,e}$")
    short_title = title.split("\n")[0]
    fig.suptitle(
        f"{short_title}: element-weighted probability-mass histograms\n{formula}\n"
        "Blue = CEMC mass inside Random support; red = CEMC mass outside",
        fontsize=19, y=1.06,
    )
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="#303030", lw=2.15,
               label="Random weighted probability mass"),
        Patch(facecolor="#e8e8e8", edgecolor="none",
              label="Random occupied-bin support"),
        Patch(facecolor="#4f88c6", edgecolor="none",
              label="CEMC mass inside Random support"),
        Patch(facecolor="#c85e64", edgecolor="none",
              label="CEMC mass outside Random support"),
        Line2D([0], [0], color="black", ls="--", lw=2.2,
               label="Optimal OH BE = 1.1 eV"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 0.89))
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows, edges


def run_histograms(model, surface, layers):
    PLOTS_1X3.mkdir(parents=True, exist_ok=True)
    PLOTS_1X5.mkdir(parents=True, exist_ok=True)
    by_temp = {}
    activity_by_temp = {}
    evaluated_by_temp = {}
    activity_rows = []
    with tempfile.TemporaryDirectory(prefix="ptpdirrhru_new_shift_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            base.evaluate(model, surface, base.reconstruct_random(run, tmpdir), SHIFTS)
            for run in range(N_RUNS)
        ]
        for temperature in base.TEMPERATURES:
            cemc = base.load_cemc(temperature)
            evaluated = {
                "Random": random_eval,
                "CEMC": [base.evaluate(model, surface, cemc[run], SHIFTS) for run in range(N_RUNS)],
                "CEMC + layer shuffle": [
                    base.evaluate(
                        model, surface,
                        base.shuffle_within_layers(
                            cemc[run], layers,
                            base.stable_seed("PtPdRhRuIr-log-KDE-peak-layer-shuffle-v1",
                                             ML_NAME, run, temperature),
                        ),
                        SHIFTS,
                    )
                    for run in range(N_RUNS)
                ],
            }
            evaluated_by_temp[temperature] = evaluated
            data = packed(evaluated)
            activity = activities(evaluated, model)
            by_temp[temperature] = data
            activity_by_temp[temperature] = activity
            for method in base.METHODS:
                for run, value in enumerate(activity[method]):
                    activity_rows.append({
                        "temperature_K": temperature, "method": method, "slab_run": run,
                        "activity_no_log": value,
                    })
            label = "final 298 K" if temperature == 298 else f"{temperature} K"
            stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
            title = (f"PtPdIrRhRu {ML_NAME} equi-atomic fcc(111): {label}\n"
                     "OH binding-energy distribution")
            plot_1x3(data, activity, title,
                     PLOTS_1X3 / f"{stem}_OH_BE_distribution_1x3.png", model)
            plot_1x5_mass(data, activity, title,
                          PLOTS_1X5 / f"{stem}_OH_BE_probability_mass_1x5.png", model)

    data_all = {
        method: (
            np.concatenate([by_temp[t][method][0] for t in base.TEMPERATURES]),
            np.concatenate([by_temp[t][method][1] for t in base.TEMPERATURES]),
        )
        for method in base.METHODS
    }
    activity_all = {
        method: np.asarray([
            np.mean([activity_by_temp[t][method][run] for t in base.TEMPERATURES])
            for run in range(N_RUNS)
        ])
        for method in base.METHODS
    }
    title_all = (f"PtPdIrRhRu {ML_NAME} equi-atomic fcc(111): all 19 temperatures\n"
                 "OH binding-energy distribution")
    plot_1x3(data_all, activity_all, title_all,
             PLOTS_1X3 / "all_temperatures_OH_BE_distribution_1x3.png", model)
    bin_rows, edges = plot_1x5_mass(
        data_all, activity_all, title_all,
        PLOTS_1X5 / "all_temperatures_OH_BE_probability_mass_1x5.png",
        model, save_bins=True,
    )
    pd.DataFrame(bin_rows).to_csv(OUT_1X5 / "all_temperatures_probability_mass_bins.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(OUT_1X3 / "activity_by_slab_temperature.csv", index=False)
    return evaluated_by_temp, data_all, activity_all, edges


def hist_by_run(values_by_run, labels_by_run, element, edges):
    presence = np.zeros((N_RUNS, len(edges) - 1), dtype=bool)
    weight = np.zeros((N_RUNS, len(edges) - 1), dtype=float)
    for run in range(N_RUNS):
        selected = values_by_run[run][labels_by_run[run] == element]
        counts, _ = np.histogram(selected, bins=edges)
        presence[run] = counts > 0
        weight[run] = counts
    return presence, weight


def bootstrap_joint(evaluated_by_temp, edges):
    OUT_BOOT.mkdir(parents=True, exist_ok=True)
    random_rows = evaluated_by_temp[base.TEMPERATURES[0]]["Random"]
    random_values = [row[0] for row in random_rows]
    random_labels = [row[1] for row in random_rows]
    cemc_values = [
        np.concatenate([evaluated_by_temp[t]["CEMC"][run][0] for t in base.TEMPERATURES])
        for run in range(N_RUNS)
    ]
    cemc_labels = [
        np.concatenate([evaluated_by_temp[t]["CEMC"][run][1] for t in base.TEMPERATURES])
        for run in range(N_RUNS)
    ]
    rng = np.random.default_rng(base.stable_seed(
        "PtPdIrRhRu-log-KDE-peak-paired-joint-bootstrap-v1", ML_NAME))
    samples = rng.integers(0, N_RUNS, size=(N_BOOTSTRAP, N_RUNS))
    rows = []
    reps = []
    for element in base.ELEMENTS:
        rs_presence, _ = hist_by_run(random_values, random_labels, element, edges)
        _, cemc_weight = hist_by_run(cemc_values, cemc_labels, element, edges)
        full_support = np.any(rs_presence, axis=0)
        full_weight = np.sum(cemc_weight, axis=0)
        point = np.sum(full_weight[full_support]) / np.sum(full_weight)
        values = np.empty(N_BOOTSTRAP, float)
        for start in range(0, N_BOOTSTRAP, 500):
            stop = min(start + 500, N_BOOTSTRAP)
            idx = samples[start:stop]
            support = np.any(rs_presence[idx], axis=1)
            weight = np.sum(cemc_weight[idx], axis=1)
            values[start:stop] = np.sum(weight * support, axis=1) / np.sum(weight, axis=1)
        reps.append(values)
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.append({
            "element": element, "full_data_estimate_percent": 100 * point,
            "paired_joint_bootstrap_mean_percent": 100 * np.mean(values),
            "paired_joint_95CI_lower_percent": 100 * lo,
            "paired_joint_95CI_upper_percent": 100 * hi,
            "lower_bound_at_least_90_percent": bool(lo >= 0.90),
            "n_slab_runs": N_RUNS, "n_bootstrap": N_BOOTSTRAP,
        })
    reps = np.vstack(reps)
    point_values = np.asarray([r["full_data_estimate_percent"] for r in rows])
    site_rep = np.mean(reps, axis=0)
    site_lo, site_hi = np.percentile(site_rep, [2.5, 97.5])
    site_row = {
        "element": "Element mean", "full_data_estimate_percent": np.mean(point_values),
        "paired_joint_bootstrap_mean_percent": 100 * np.mean(site_rep),
        "paired_joint_95CI_lower_percent": 100 * site_lo,
        "paired_joint_95CI_upper_percent": 100 * site_hi,
        "lower_bound_at_least_90_percent": bool(site_lo >= 0.90),
        "n_slab_runs": N_RUNS, "n_bootstrap": N_BOOTSTRAP,
    }
    frame = pd.DataFrame(rows + [site_row])
    frame.to_csv(OUT_BOOT / "paired_joint_bootstrap_95CI.csv", index=False)
    np.savez_compressed(
        OUT_BOOT / "paired_joint_bootstrap_replicates.npz",
        element_replicates_percent=100 * reps,
        element_mean_replicates_percent=100 * site_rep,
        elements=np.asarray(base.ELEMENTS),
    )
    plot_bootstrap(frame, 100 * reps, 100 * site_rep)
    return frame


def plot_bootstrap(frame, reps, site_rep):
    colors = base.ELEMENT_COLORS
    all_rep = np.concatenate([reps.ravel(), site_rep])
    xmin = max(0.0, np.floor(np.percentile(all_rep, 0.05) / 5) * 5 - 2.5)
    xmax = 100.5
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    fig.subplots_adjust(left=0.14, right=0.72, top=0.80, bottom=0.16)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.65, 5.7)
    ax.invert_yaxis()
    ax.axvspan(90, xmax, color="#2ca02c", alpha=0.045)
    ax.axvline(90, color="black", ls="--", lw=2.0)
    ax.text(90.3, -0.38, "90% criterion", fontsize=16, ha="left", va="center")
    for y, element in enumerate(base.ELEMENTS):
        row = frame.iloc[y]
        values = reps[y]
        lo0, hi0 = np.percentile(values, [0.1, 99.9])
        pad = max(0.15, 0.35 * (hi0 - lo0))
        x = np.linspace(max(xmin, lo0 - pad), min(xmax, hi0 + pad), 220)
        d = gaussian_kde(values)(x); d /= d.max()
        color = colors[element]
        ax.fill_between(x, y, y + 0.34 * d, color=color, alpha=0.28, linewidth=0)
        ax.plot(x, y + 0.34 * d, color=color, lw=1.2)
        lo = row["paired_joint_95CI_lower_percent"]
        hi = row["paired_joint_95CI_upper_percent"]
        point = row["full_data_estimate_percent"]
        mean = row["paired_joint_bootstrap_mean_percent"]
        ax.hlines(y, lo, hi, color=color, lw=3.4)
        ax.vlines([lo, hi], y - 0.13, y + 0.13, color=color, lw=1.6)
        ax.scatter(point, y, s=100, color=color, edgecolor="white", lw=0.9, zorder=5)
        ax.vlines(mean, y - 0.24, y + 0.24, color=color, lw=1.7)
        passed = bool(row["lower_bound_at_least_90_percent"])
        ax.text(xmax + 0.35, y, f"{point:5.2f}  [{lo:5.2f}, {hi:5.2f}]  {'✓' if passed else '×'}",
                fontsize=16, ha="left", va="center",
                color="#1b7837" if passed else "#b2182b", clip_on=False)
    y = 5.25
    row = frame.iloc[-1]
    lo0, hi0 = np.percentile(site_rep, [0.1, 99.9])
    pad = max(0.15, 0.35 * (hi0 - lo0))
    x = np.linspace(max(xmin, lo0 - pad), min(xmax, hi0 + pad), 220)
    d = gaussian_kde(site_rep)(x); d /= d.max()
    ax.fill_between(x, y, y + 0.40 * d, color="#333333", alpha=0.20, linewidth=0)
    ax.plot(x, y + 0.40 * d, color="#333333", lw=1.2)
    lo, hi = row["paired_joint_95CI_lower_percent"], row["paired_joint_95CI_upper_percent"]
    point, mean = row["full_data_estimate_percent"], row["paired_joint_bootstrap_mean_percent"]
    ax.hlines(y, lo, hi, color="#333333", lw=4.4)
    ax.vlines([lo, hi], y - 0.14, y + 0.14, color="#333333", lw=1.7)
    ax.scatter(point, y, s=145, color="#333333", marker="D", edgecolor="white", lw=0.9, zorder=5)
    ax.vlines(mean, y - 0.27, y + 0.27, color="#333333", lw=1.8)
    passed = bool(row["lower_bound_at_least_90_percent"])
    ax.text(xmax + 0.35, y, f"{point:5.2f}  [{lo:5.2f}, {hi:5.2f}]  {'✓' if passed else '×'}",
            fontsize=16, fontweight="bold", ha="left", va="center",
            color="#1b7837" if passed else "#b2182b", clip_on=False)
    ax.axhline(4.65, color="black", lw=1.2)
    ax.set_yticks([0, 1, 2, 3, 4, y], [*base.ELEMENTS, "Element mean"], fontsize=17)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#d0d0d0", lw=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("CEMC probability mass contained in resampled Random occupied-bin support (%)",
                  fontsize=18)
    ax.set_ylabel("Top-site element", fontsize=18)
    ax.text(xmax + 0.35, -0.38, "Estimate  [95% CI]  pass", fontsize=16,
            ha="left", va="center", fontweight="bold", clip_on=False)
    fig.suptitle(f"PtPdIrRhRu {ML_NAME} fcc(111): paired-joint slab-bootstrap containment",
                 fontsize=22, fontweight="bold", y=0.965)
    ax.set_title(
        "Random support and CEMC mass jointly resampled by paired slab run; 20 runs, "
        "10,000 replicates; percentile 95% CI\n"
        "Ridge = bootstrap distribution, circle = full-sample containment estimate, "
        "vertical tick = bootstrap mean, bar = 95% CI",
        fontsize=15.2, pad=22,
    )
    fig.savefig(OUT_BOOT / "all_temperatures_paired_joint_bootstrap_ridgeline_forest.png",
                dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", required=True, choices=[f"ML{i}" for i in range(1, 7)])
    args = parser.parse_args()
    selected = configure(args.ml)
    for out in (OUT_1X3, OUT_1X5, OUT_BOOT):
        out.mkdir(parents=True, exist_ok=True)
    model_path = FIT_ROOT / "hea_uq_cemc_cpp_v1.0" / f"uq_{ML_NAME}_full" / "activity_model_oh.txt"
    model = base.parse_model(model_path)
    surface, layers = base.surface_sites_and_validate(model)
    if abs(model["e_opt"] - 1.1) > 1e-12:
        raise ValueError(f"Expected optimum 1.1 eV, found {model['e_opt']}")
    evaluated_by_temp, data_all, activity_all, edges = run_histograms(model, surface, layers)
    boot = bootstrap_joint(evaluated_by_temp, edges)
    metadata = {
        "system": "PtPdIrRhRu", "ML": ML_NAME, "composition": "equi-atomic", "facet": "fcc(111)",
        "adsorption_site": "OH top site", "new_BE_shifts_eV": SHIFTS,
        "BE_shift_source": str(BE_TABLE),
        "composition_shift_source": str(COMP_TABLE),
        "representative_selection": "maximum 3D-histogram KDE probability for CEMC log-activity with tau > 0",
        "representative_trial": int(selected["trial"]),
        "representative_temperature_K": float(selected["annealing_temperature_K"]),
        "representative_tau": float(selected["tau"]),
        "representative_mse": float(selected["mse"]),
        "representative_crps": float(selected["crps"]),
        "composition_shift_applied_to_equi_atomic_slabs": False,
        "activity_model": str(model_path),
        "shift_formula": "final_OH_BE = linear_model_OH_BE - BE_shift(top-site element)",
        "optimal_OH_BE_eV": 1.1, "methods": list(base.METHODS),
        "one_by_five_methods": ["Random", "CEMC"],
        "one_by_five_style": (
            "Random black step and gray occupied-bin support; CEMC probability mass "
            "blue inside and red outside Random support; layer shuffle excluded"
        ),
        "temperatures_K": list(base.TEMPERATURES), "n_slab_runs": N_RUNS,
        "n_bins_probability_mass": N_BINS,
        "probability_mass_normalization": "within each method and top-site element; sum over bins = 1",
        "bootstrap": "10,000-replicate paired-joint slab-run cluster percentile bootstrap",
        "bootstrap_target": "CEMC probability mass in resampled Random occupied-bin support",
        "n_png_1x3": len(list(PLOTS_1X3.glob("*.png"))),
        "n_png_1x5": len(list(PLOTS_1X5.glob("*.png"))),
        "n_png_bootstrap": len(list(OUT_BOOT.glob("*.png"))),
        "n_pdf": 0,
        "paired_joint_element_mean_full_estimate_percent": float(boot.iloc[-1]["full_data_estimate_percent"]),
        "paired_joint_element_mean_95CI_percent": [
            float(boot.iloc[-1]["paired_joint_95CI_lower_percent"]),
            float(boot.iloc[-1]["paired_joint_95CI_upper_percent"]),
        ],
    }
    for out in (OUT_1X3, OUT_1X5, OUT_BOOT):
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
