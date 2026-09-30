#!/usr/bin/env python3
"""Quantify all-temperature CEMC containment in Random binary bin support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import analyze_hbe_distributions_new_be_shift as analysis


OUTPUT_CSV = analysis.BASE / "CEMC_in_Random_binary_bin_containment_all_temperatures.csv"
OUTPUT_JSON = analysis.BASE / "CEMC_in_Random_binary_bin_containment_all_temperatures.json"
HISTOGRAM_DIRNAME = (
    "hbe_distribution_analysis_log_activity_top100_new_BE_shift_"
    "probability_mass_weighted_histogram_1x5"
)


def evaluate_case(facet: str, site: str, case: str) -> list[dict]:
    combo = f"{facet}/{site}"
    workdir = analysis.BASE / facet / site
    model = analysis.parse_model(workdir / "inputs/activity_model.txt")
    groups = analysis.layer_groups(workdir / "inputs/template.cif")
    _, shifts, _ = analysis.load_shift_data(case)
    pooled = {method: {"values": [], "fractions": []} for method in analysis.METHODS}

    with tempfile.TemporaryDirectory(prefix=f"containment_{facet}_{site}_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            analysis.evaluate_sites(
                model, analysis.reconstruct_random(workdir, run, tmpdir), shifts)
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(workdir, temperature)
            evaluated = {"Random": random_eval, "CEMC": [], "CEMC + layer shuffle": []}
            for run in range(20):
                occ = cemc[run]
                evaluated["CEMC"].append(analysis.evaluate_sites(model, occ, shifts))
                shuffled = analysis.shuffle_occ(
                    occ, groups,
                    analysis.stable_seed(
                        "FeCoNiPdPt-equi-layer-shuffle-v1", combo, run, temperature))
                evaluated["CEMC + layer shuffle"].append(
                    analysis.evaluate_sites(model, shuffled, shifts))
            for method in analysis.METHODS:
                pooled[method]["values"].append(
                    np.concatenate([result[2] - model.e_opt for result in evaluated[method]]))
                pooled[method]["fractions"].append(
                    np.vstack([result[3] for result in evaluated[method]]))

    for method in analysis.METHODS:
        pooled[method]["values"] = np.concatenate(pooled[method]["values"])
        pooled[method]["fractions"] = np.vstack(pooled[method]["fractions"])

    combined = np.concatenate([pooled[m]["values"] for m in analysis.METHODS])
    lo = min(float(combined.min()), 0.0) - 0.06
    hi = max(float(combined.max()), 0.0) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    rows = []
    panel_data = []
    bin_rows = []
    for ei, element in enumerate(analysis.ELEMENTS):
        method_counts = {}
        method_weight = {}
        for method in ("Random", "CEMC"):
            values = pooled[method]["values"]
            weights = pooled[method]["fractions"][:, ei]
            mask = weights > 0.0
            method_counts[method], _ = np.histogram(values[mask], bins=edges)
            method_weight[method], _ = np.histogram(
                values[mask], bins=edges, weights=weights[mask])

        random_support = method_counts["Random"] > 0
        cemc_support = method_counts["CEMC"] > 0
        intersection = random_support & cemc_support
        n_cemc = int(cemc_support.sum())
        n_intersection = int(intersection.sum())
        n_outside = int((cemc_support & ~random_support).sum())
        binary_containment = n_intersection / n_cemc if n_cemc else float("nan")
        cemc_total_weight = float(method_weight["CEMC"].sum())
        cemc_weight_inside = float(method_weight["CEMC"][random_support].sum())
        mass_containment = (cemc_weight_inside / cemc_total_weight
                            if cemc_total_weight > 0.0 else float("nan"))
        random_total_weight = float(method_weight["Random"].sum())
        random_probability = (method_weight["Random"] / random_total_weight
                              if random_total_weight > 0.0
                              else np.zeros_like(method_weight["Random"], dtype=float))
        cemc_probability = (method_weight["CEMC"] / cemc_total_weight
                            if cemc_total_weight > 0.0
                            else np.zeros_like(method_weight["CEMC"], dtype=float))
        cemc_inside = np.where(random_support, cemc_probability, 0.0)
        cemc_outside = np.where(~random_support, cemc_probability, 0.0)
        panel_data.append({
            "element": element,
            "random_support": random_support,
            "random_probability": random_probability,
            "cemc_inside": cemc_inside,
            "cemc_outside": cemc_outside,
            "mass_containment": mass_containment,
        })
        for bi in range(len(centers)):
            bin_rows.append({
                "case": case, "facet": facet, "site": site, "element": element,
                "bin_index": bi, "bin_left_eV": edges[bi],
                "bin_center_eV": centers[bi], "bin_right_eV": edges[bi + 1],
                "random_occupied": bool(random_support[bi]),
                "random_fractional_probability_mass": random_probability[bi],
                "cemc_fractional_probability_mass": cemc_probability[bi],
                "cemc_mass_inside_random_support": cemc_inside[bi],
                "cemc_mass_outside_random_support": cemc_outside[bi],
            })
        rows.append({
            "facet": facet,
            "site": site,
            "case": case,
            "element": element,
            "n_bins": len(edges) - 1,
            "bin_width_eV": float(edges[1] - edges[0]),
            "random_occupied_bins": int(random_support.sum()),
            "cemc_occupied_bins": n_cemc,
            "shared_occupied_bins": n_intersection,
            "cemc_occupied_bins_outside_random": n_outside,
            "binary_support_containment_fraction": binary_containment,
            "binary_support_containment_percent": 100.0 * binary_containment,
            "cemc_fractional_probability_mass_in_random_support_fraction": mass_containment,
            "cemc_fractional_probability_mass_in_random_support_percent": 100.0 * mass_containment,
            "binary_support_at_least_90_percent": bool(binary_containment >= 0.90),
            "probability_mass_at_least_90_percent": bool(mass_containment >= 0.90),
        })

    output_dir = workdir / HISTOGRAM_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    })
    fig, axes = plt.subplots(1, 5, figsize=(25, 7.0), sharex=True, sharey=True)
    for ax, data in zip(axes, panel_data):
        ax.fill_between(
            centers, 0.0, 1.0, where=data["random_support"], step="mid",
            transform=ax.get_xaxis_transform(), color="#777777", alpha=0.06)
        ax.bar(centers, data["cemc_inside"], width=0.92 * width,
               color="#2468B4", alpha=0.72, linewidth=0.0)
        ax.bar(centers, data["cemc_outside"], width=0.92 * width,
               color="#C44E52", alpha=0.82, linewidth=0.0)
        ax.step(centers, data["random_probability"], where="mid",
                color="#333333", lw=2.0)
        ax.axvline(0.0, color="black", ls="--", lw=1.8)
        ax.set_title(data["element"], pad=8)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.grid(alpha=0.18)
        ax.text(
            0.98, 0.95,
            fr"$C_{{\mathrm{{{data['element']}}}}}$ = {100.0 * data['mass_containment']:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=16)
    axes[0].set_ylabel("Fractional probability mass per bin")
    formula = (
        r"$p_{\mathrm{CEMC},b,e}=\sum_{i\in b}w_{i,e}/\sum_i w_{i,e}$;  "
        r"$C_e=\sum_b I_{\mathrm{RS},b,e}\,p_{\mathrm{CEMC},b,e}$"
    )
    fig.suptitle(
        f"FeCoNiPdPt fcc({facet}) {site}: all-temperature raw weighted histograms\n"
        f"{formula}\nBlue = CEMC mass inside Random support; red = CEMC mass outside",
        fontsize=18, y=1.10)
    legend = [
        Line2D([0], [0], color="#333333", lw=2.0,
               label="Random weighted probability mass"),
        Patch(facecolor="#777777", alpha=0.12, label="Random occupied-bin support"),
        Patch(facecolor="#2468B4", alpha=0.72, label="CEMC mass inside Random support"),
        Patch(facecolor="#C44E52", alpha=0.82, label="CEMC mass outside Random support"),
        Line2D([0], [0], color="black", ls="--", lw=1.8,
               label=r"Optimal $\Delta G_{\mathrm{H}}$"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure_path = output_dir / "all_temperatures_probability_mass_weighted_histograms.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(bin_rows).to_csv(output_dir / "weighted_histogram_bin_data.csv", index=False)
    metadata = {
        "case": case, "facet": facet, "site": site, "n_bins": 90,
        "figure": str(figure_path), "smoothing_applied": False,
        "cemc_probability_formula": "sum(w_i,e in bin b) / sum(w_i,e over all bins)",
        "random_support_formula": "1 if any Random ensemble with w_i,e > 0 is in bin b, else 0",
        "containment_formula": "sum(CEMC probability mass in Random-occupied bins)",
        "existing_graphs_preserved": True, "pdf_created": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main() -> None:
    rows = []
    for facet, site, case in analysis.COMBOS:
        rows.extend(evaluate_case(facet, site, case))
        print(f"Ready {case}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_CSV, index=False)
    summary = {
        "comparison": "CEMC contained in Random all-temperature binary bin support",
        "n_cases": len(analysis.COMBOS),
        "n_case_element_pairs": len(frame),
        "n_weighted_histogram_png": len(analysis.COMBOS),
        "weighted_histogram_directory_name": HISTOGRAM_DIRNAME,
        "n_bins_per_pair": 90,
        "binary_support_containment_definition": (
            "number of CEMC-occupied bins also occupied by Random divided by "
            "number of CEMC-occupied bins"
        ),
        "probability_mass_containment_definition": (
            "CEMC fractional elemental histogram weight in Random-occupied bins divided "
            "by total CEMC fractional elemental histogram weight"
        ),
        "pairs_binary_support_at_least_90_percent": int(
            frame["binary_support_at_least_90_percent"].sum()),
        "pairs_probability_mass_at_least_90_percent": int(
            frame["probability_mass_at_least_90_percent"].sum()),
        "all_pairs_binary_support_at_least_90_percent": bool(
            frame["binary_support_at_least_90_percent"].all()),
        "all_pairs_probability_mass_at_least_90_percent": bool(
            frame["probability_mass_at_least_90_percent"].all()),
        "minimum_binary_support_containment_percent": float(
            frame["binary_support_containment_percent"].min()),
        "minimum_probability_mass_containment_percent": float(
            frame["cemc_fractional_probability_mass_in_random_support_percent"].min()),
        "mean_binary_support_containment_percent": float(
            frame["binary_support_containment_percent"].mean()),
        "mean_probability_mass_containment_percent": float(
            frame["cemc_fractional_probability_mass_in_random_support_percent"].mean()),
        "below_90_percent": frame.loc[
            ~(frame["binary_support_at_least_90_percent"] &
              frame["probability_mass_at_least_90_percent"]),
            ["case", "element", "binary_support_containment_percent",
             "cemc_fractional_probability_mass_in_random_support_percent"]
        ].to_dict(orient="records"),
        "output_csv": str(OUTPUT_CSV),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
