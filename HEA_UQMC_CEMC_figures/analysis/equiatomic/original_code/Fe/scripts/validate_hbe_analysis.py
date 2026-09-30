#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
COMBOS = (("100", "bridge"), ("100", "hollow"), ("110", "bridge"), ("111", "hollow"), ("111", "top"))
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
KB_EV_PER_K = 8.617333262145e-5
expected_denominator = {("100", "bridge"): 2, ("100", "hollow"): 4,
                        ("110", "bridge"): 2, ("111", "hollow"): 3,
                        ("111", "top"): 1}
result = {"combinations": {}, "validated": False}
total_site_rows = 0

for facet, site in COMBOS:
    root = BASE / facet / site / "hbe_distribution_analysis"
    pngs = sorted((root / "plots_by_temperature").glob("*.png"))
    gzfiles = sorted((root / "site_data_by_temperature").glob("*.csv.gz"))
    pdfs = sorted(root.glob("*.pdf"))
    assert len(pngs) == 20 and len(gzfiles) == 19 and len(pdfs) == 0
    metadata = json.loads((root / "analysis_metadata.json").read_text())
    assert metadata["composition_shift_applied"] is False
    assert metadata["deltaG_H_formula"] == "final_H_BE_eV - reference_optimal_H_BE_eV"
    assert abs(float(metadata["optimal_deltaG_H_eV"])) < 1e-12
    reference_optimal_hbe = float(metadata["reference_optimal_H_BE_eV"])
    assert metadata["activity_formula_no_log"] == "mean(exp(-abs(deltaG_H)/(k_B*T_activity)))"
    activity_temperature = float(metadata["activity_temperature_K"])
    composition = metadata["composition_unshifted"]
    assert set(composition) == set(ELEMENTS)
    assert all(abs(float(composition[e]) - 0.2) < 1e-12 for e in ELEMENTS)
    assert abs(float(metadata["composition_sum"]) - 1.0) < 1e-12
    n_activity_sites = int(metadata["n_activity_sites_per_slab"])
    expected_rows_per_file = 3 * 20 * n_activity_sites
    nrows = 0
    max_formula_error = 0.0
    max_fraction_sum_error = 0.0
    max_fraction_grid_error = 0.0
    max_delta_g_mean_error = 0.0
    max_activity_error = 0.0
    weighted_totals = {}
    activity_totals = {}
    all_temperature_activity_totals = {}
    denominator = expected_denominator[(facet, site)]
    for path in gzfiles:
        rows_here = 0
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_here += 1
                linear = float(row["linear_H_BE_eV"])
                shift = float(row["site_BE_shift_eV"])
                final = float(row["final_H_BE_eV"])
                fractions = [float(row[f"zone1_fraction_{e}"]) for e in ELEMENTS]
                temperature = int(row["temperature_K"])
                method = row["method"]
                site_activity = math.exp(-abs(final - reference_optimal_hbe) /
                                         (KB_EV_PER_K * activity_temperature))
                activity_total = activity_totals.setdefault((temperature, method), [0.0, 0])
                activity_total[0] += site_activity
                activity_total[1] += 1
                all_total = all_temperature_activity_totals.setdefault(method, [0.0, 0])
                all_total[0] += site_activity
                all_total[1] += 1
                for element, fraction in zip(ELEMENTS, fractions):
                    totals = weighted_totals.setdefault((temperature, method, element), [0.0, 0.0])
                    totals[0] += final * fraction
                    totals[1] += fraction
                max_formula_error = max(max_formula_error, abs(final - (linear - shift)))
                max_fraction_sum_error = max(max_fraction_sum_error, abs(sum(fractions) - 1.0))
                max_fraction_grid_error = max(max_fraction_grid_error,
                                              max(abs(x * denominator - round(x * denominator)) for x in fractions))
        assert rows_here == expected_rows_per_file, (path, rows_here, expected_rows_per_file)
        nrows += rows_here
    assert max_formula_error < 1e-10
    assert max_fraction_sum_error < 1e-10
    assert max_fraction_grid_error < 1e-10
    with (root / "element_HBE_distribution_summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    summary_rows = len(summary)
    assert summary_rows == 285
    for row in summary:
        key = (int(row["temperature_K"]), row["method"], row["element"])
        weighted_sum, total_weight = weighted_totals[key]
        actual_text = row["weighted_mean_deltaG_H_eV"]
        if total_weight > 0.0:
            expected = weighted_sum / total_weight - reference_optimal_hbe
            error = abs(float(actual_text) - expected)
            max_delta_g_mean_error = max(max_delta_g_mean_error, error)
        else:
            assert actual_text == "" or math.isnan(float(actual_text))
        activity_sum, activity_count = activity_totals[(key[0], key[1])]
        activity_error = abs(float(row["activity_no_log"]) - activity_sum / activity_count)
        max_activity_error = max(max_activity_error, activity_error)
    assert max_delta_g_mean_error < 1e-10
    with (root / "all_temperatures_average_element_deltaG_H_summary.csv").open(newline="") as handle:
        averaged_summary = list(csv.DictReader(handle))
    assert len(averaged_summary) == 15
    for row in averaged_summary:
        activity_sum, activity_count = all_temperature_activity_totals[row["method"]]
        activity_error = abs(float(row["activity_no_log"]) - activity_sum / activity_count)
        max_activity_error = max(max_activity_error, activity_error)
    assert max_activity_error < 1e-10
    result["combinations"][f"fcc{facet}_{site}"] = {
        "png_files": len(pngs), "multipage_pdf_files": len(pdfs),
        "compressed_site_data_files": len(gzfiles), "site_data_rows": nrows,
        "summary_rows": summary_rows, "n_activity_sites_per_slab": n_activity_sites,
        "rows_per_temperature_file": expected_rows_per_file,
        "zone1_denominator": denominator,
        "max_final_BE_formula_error_eV": max_formula_error,
        "max_zone1_fraction_sum_error": max_fraction_sum_error,
        "max_zone1_fraction_grid_error": max_fraction_grid_error,
        "reference_optimal_H_BE_eV": reference_optimal_hbe,
        "optimal_deltaG_H_eV": 0.0,
        "max_weighted_mean_deltaG_H_error_eV": max_delta_g_mean_error,
        "max_no_log_activity_error": max_activity_error,
        "activity_temperature_K": activity_temperature,
        "all_temperatures_average_summary_rows": len(averaged_summary),
        "composition_shift_applied": False,
        "composition_ratio_each_element": 0.2,
    }
    total_site_rows += nrows

assert (BASE / "H_BE_DISTRIBUTION_ANALYSIS_COMPLETE").exists()
result.update({"total_png_files": 100, "total_multipage_pdf_files": 0,
               "total_compressed_site_data_files": 95, "total_site_data_rows": total_site_rows,
               "total_summary_rows": 1425, "validated": True})
(BASE / "H_BE_distribution_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
