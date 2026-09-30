#!/usr/bin/env python3
"""Quantify all-temperature CEMC containment in Random binary bin support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_hbe_distributions_new_be_shift as analysis


OUTPUT_CSV = analysis.BASE / "CEMC_in_Random_binary_bin_containment_all_temperatures.csv"
OUTPUT_JSON = analysis.BASE / "CEMC_in_Random_binary_bin_containment_all_temperatures.json"


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
    rows = []
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
