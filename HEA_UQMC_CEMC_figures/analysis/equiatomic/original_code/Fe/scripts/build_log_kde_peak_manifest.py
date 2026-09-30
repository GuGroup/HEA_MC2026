#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
FIT = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/fit_exp/log_activity_joint_probability_tsne_activity_maps")
OUT = BASE / "log_activity_KDE_peak_trial_analysis_manifest"
CASES = [
    ("100", "bridge", "fcc100_bridge"),
    ("100", "hollow", "fcc100_hollow"),
    ("110", "bridge", "fcc110_bridge"),
    ("111", "hollow", "fcc111_hollow"),
    ("111", "top", "fcc111_top"),
]

OUT.mkdir(parents=True, exist_ok=True)
be_rows, comp_rows, inventory = [], [], []
for facet, site, case in CASES:
    source = FIT / case / "kde_smoothed"
    be = pd.read_csv(source / "log_kde_representative_trial_BE_shifts.csv")
    comp = pd.read_csv(source / "log_kde_representative_trial_composition_shifts.csv")
    be_rows.append(be)
    comp_rows.append(comp)
    work = BASE / facet / site
    one3 = work / "hbe_distribution_analysis_log_activity_KDE_peak_trial_BE_shift_1x3"
    one5 = work / "hbe_distribution_analysis_log_activity_KDE_peak_trial_BE_shift_probability_mass_weighted_histogram_1x5"
    boot = BASE / "log_activity_KDE_peak_trial_paired_joint_bootstrap" / "by_facet_site" / f"{case}_joint_bootstrap_ridgeline_forest.png"
    rec = {
        "case": case,
        "facet": f"fcc({facet})",
        "site": site,
        "representative_trial": int(be.iloc[0]["trial"]),
        "representative_temperature_K": float(be.iloc[0]["annealing_temperature_K"]),
        "tau": float(be.iloc[0]["tau"]),
        "mse": float(be.iloc[0]["mse"]),
        "crps": float(be.iloc[0]["crps"]),
        "BE_shift_source": str(source / "log_kde_representative_trial_BE_shifts.csv"),
        "composition_shift_source": str(source / "log_kde_representative_trial_composition_shifts.csv"),
        "one_by_three_directory": str(one3),
        "one_by_three_png_count": len(list((one3 / "plots_by_temperature").glob("*.png"))),
        "one_by_five_figure": str(one5 / "all_temperatures_probability_mass_weighted_histograms.png"),
        "one_by_five_png_count": len(list(one5.glob("*.png"))),
        "bootstrap_figure": str(boot),
        "bootstrap_exists": boot.exists(),
        "new_pdf_count": len(list(one3.rglob("*.pdf"))) + len(list(one5.rglob("*.pdf"))),
    }
    inventory.append(rec)

pd.concat(be_rows, ignore_index=True).to_csv(OUT / "selected_BE_shifts_all_facets_sites.csv", index=False)
pd.concat(comp_rows, ignore_index=True).to_csv(OUT / "selected_composition_shifts_all_facets_sites.csv", index=False)
pd.DataFrame(inventory).to_csv(OUT / "result_inventory.csv", index=False)
summary = {
    "system": "FeCoNiPdPt",
    "selection": "single CEMC log-activity trial in the maximum KDE-smoothed (tau, MSE, CRPS) 3-D histogram bin with tau > 0",
    "BE_shift_applied_to_BE_distributions": True,
    "composition_shift_applied_to_equi_atomic_slabs": False,
    "n_facet_site_cases": len(CASES),
    "all_cases_have_20_one_by_three_png": all(r["one_by_three_png_count"] == 20 for r in inventory),
    "all_cases_have_one_one_by_five_png": all(r["one_by_five_png_count"] == 1 for r in inventory),
    "all_cases_have_bootstrap_png": all(r["bootstrap_exists"] for r in inventory),
    "total_new_pdf_count": sum(r["new_pdf_count"] for r in inventory),
    "inventory": inventory,
}
(OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "inventory"}, indent=2))
