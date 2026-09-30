#!/usr/bin/env python3
"""Collect current log-KDE representative trials and their exact BE shifts."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIT = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/fit_exp")
OUT = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/new_BE_shift_completed_sites_log_KDE_peak")
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
CASES = [
    {
        "case": "fcc100_bridge", "label": "fcc(100) bridge",
        "selection": FIT / "uqmc_new_BE_shift_fcc_100/individual_slab_joint_probability_tsne_analysis/bridge/log_activity/selected_kde_representative_trials.csv",
        "be": FIT / "uqmc_new_BE_shift_fcc_100/bridge/results/be/be_shifts.csv",
    },
    {
        "case": "fcc100_hollow", "label": "fcc(100) hollow",
        "selection": FIT / "uqmc_new_BE_shift_fcc_100/individual_slab_joint_probability_tsne_analysis/hollow/log_activity/selected_kde_representative_trials.csv",
        "be": FIT / "uqmc_new_BE_shift_fcc_100/hollow/results/be/be_shifts.csv",
    },
    {
        "case": "fcc110_bridge", "label": "fcc(110) bridge",
        "selection": FIT / "uqmc_new_BE_shift_fcc_110/individual_slab_joint_probability_tsne_analysis/bridge/log_activity/selected_kde_representative_trials.csv",
        "be": FIT / "uqmc_new_BE_shift_fcc_110/bridge/results/be/be_shifts.csv",
    },
    {
        "case": "fcc111_hollow", "label": "fcc(111) hollow",
        "selection": FIT / "uqmc_new_BE_shift_fcc_111/individual_slab_joint_probability_tsne_analysis/fcc/log_activity/selected_kde_representative_trials.csv",
        "be": FIT / "uqmc_new_BE_shift_fcc_111/fcc/results/be/be_shifts.csv",
    },
    {
        "case": "fcc111_top", "label": "fcc(111) top",
        "selection": FIT / "uqmc_new_BE_shift_fcc_111/individual_slab_joint_probability_tsne_analysis/top/log_activity/selected_kde_representative_trials.csv",
        "be": FIT / "uqmc_new_BE_shift_fcc_111/top/results/be/be_shifts.csv",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in CASES:
        selected = pd.read_csv(spec["selection"]).iloc[0]
        trial = int(selected["cemc_trial"])
        shifts = pd.read_csv(spec["be"])
        hit = shifts[shifts["trial"] == trial]
        if len(hit) != 1:
            raise ValueError(f"{spec['case']}: expected one BE-shift row for trial {trial}, got {len(hit)}")
        hit = hit.iloc[0]
        row = {
            "system": "FeCoNiPdPt", "case": spec["case"], "facet_site": spec["label"],
            "activity_basis": "log_activity_individual_slab_ln_then_mean",
            "representative_selection": "maximum_3D_histogram_KDE_probability_CEMC",
            "trial": trial, "annealing_temperature_K": float(selected["cemc_temperature_K"]),
            "tau": float(selected["cemc_tau"]), "mse": float(selected["cemc_mse"]),
            "crps": float(selected["cemc_crps"]),
            "kde_peak_bin_probability": float(selected["cemc_peak_probability"]),
            "kde_peak_bin_raw_count": int(selected["cemc_peak_raw_count"]),
        }
        for element in ELEMENTS:
            value = float(hit[f"be_shift_{element}"])
            row[f"be_shift_{element}_exact_eV"] = value
            row[f"be_shift_{element}_rounded_2dp_eV"] = round(value, 2)
        rows.append(row)
        source = OUT / "selected_shift_sources" / spec["case"] / "kde_smoothed"
        source.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(source / "log_kde_representative_trial_BE_shifts.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "selected_BE_shifts_completed_facets_sites.csv", index=False)
    display = pd.DataFrame({
        "Facet/site": frame["facet_site"],
        "Trial": frame["trial"].astype(int),
        "T (K)": frame["annealing_temperature_K"].astype(int),
        **{element: frame[f"be_shift_{element}_exact_eV"].map(lambda x: f"{x:.6f}") for element in ELEMENTS},
    })
    display.to_csv(OUT / "selected_BE_shifts_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.axis("off")
    ax.set_title("FeCoNiPdPt BE shifts\nunit: eV", loc="left", fontsize=15, pad=14)
    table = ax.table(cellText=display.values, colLabels=display.columns,
                     cellLoc="center", colLoc="center", loc="center",
                     colWidths=[0.19, 0.08, 0.08, 0.11, 0.11, 0.11, 0.11, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f3f3f3")
        elif c == 0:
            cell.set_text_props(ha="left")
    fig.savefig(OUT / "selected_BE_shifts_table.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (OUT / "selection_manifest.json").write_text(json.dumps({
        "selection_mode": "log_activity",
        "log_definition": "natural log of each individual slab activity, then mean over 20 slabs",
        "selection": "observed CEMC trial in maximum KDE-smoothed 3D histogram bin",
        "composition": {element: 0.2 for element in ELEMENTS},
        "composition_shift_applied": False,
        "completed_cases": [row["case"] for row in rows],
        "excluded_incomplete_case": None,
        "records": rows,
    }, indent=2) + "\n")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
