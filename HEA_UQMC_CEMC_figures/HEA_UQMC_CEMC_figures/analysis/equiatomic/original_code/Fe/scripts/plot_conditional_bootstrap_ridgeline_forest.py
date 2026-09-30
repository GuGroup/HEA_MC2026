#!/usr/bin/env python3
"""One-page conditional slab-bootstrap ridgeline/forest summary."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import analyze_hbe_distributions_new_be_shift as analysis


PAIR_CSV = analysis.BASE / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_by_case_element.csv"
CASE_CSV = analysis.BASE / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_by_case.csv"
SUMMARY_JSON = analysis.BASE / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_summary.json"
REPLICATES = analysis.BASE / "CEMC_in_Random_probability_mass_slab_bootstrap_conditional_replicates.npz"
OUTDIR = analysis.BASE / "conditional_probability_mass_bootstrap_ridgeline_forest"
OUTPNG = OUTDIR / "conditional_bootstrap_all_case_element_ridgeline_forest.png"
OUTMETA = OUTDIR / "metadata.json"

CASE_LABELS = {
    "fcc100_bridge": "fcc(100) bridge",
    "fcc100_hollow": "fcc(100) hollow",
    "fcc110_bridge": "fcc(110) bridge",
    "fcc111_hollow": "fcc(111) hollow",
    "fcc111_top": "fcc(111) top",
}
COLORS = {
    "Fe": "#d95f02", "Co": "#7570b3", "Ni": "#1b9e77",
    "Pd": "#e7298a", "Pt": "#66a61e",
}


def density_xy(samples, xmin=70.0, xmax=100.2):
    lo, hi = np.percentile(samples, [0.1, 99.9])
    pad = max(0.12, 0.35 * (hi - lo))
    x = np.linspace(max(xmin, lo - pad), min(xmax, hi + pad), 180)
    y = gaussian_kde(samples)(x)
    return x, y / y.max()


def draw_ridge(ax, y0, samples, color, height=0.34, alpha=0.28):
    x, d = density_xy(samples)
    ax.fill_between(x, y0, y0 + height * d, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y0 + height * d, color=color, linewidth=1.1)


def main():
    pair = pd.read_csv(PAIR_CSV)
    case = pd.read_csv(CASE_CSV)
    summary = json.loads(SUMMARY_JSON.read_text())
    reps = np.load(REPLICATES)
    pair_rep = reps["pair_conditional_percent"]
    case_rep = reps["case_conditional_percent"]
    overall_rep = reps["overall_conditional_percent"]

    rows = []
    pair_idx = 0
    for case_idx, case_name in enumerate(case["case"]):
        for element in analysis.ELEMENTS:
            rec = pair.iloc[pair_idx]
            rows.append(("element", case_name, element, rec, pair_rep[pair_idx]))
            pair_idx += 1
        rows.append(("case", case_name, "Mean", case.iloc[case_idx], case_rep[case_idx]))
    overall = summary["overall"]
    rows.append(("overall", "all", "Overall", overall, overall_rep))

    y_positions = []
    cursor = 0.0
    for kind, *_ in rows:
        y_positions.append(cursor)
        cursor += 1.0
        if kind == "case":
            cursor += 0.65
    n = len(rows)
    fig, ax = plt.subplots(figsize=(18, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.225, right=0.82, top=0.925, bottom=0.065)
    ax.set_xlim(70.0, 100.5)
    ax.set_ylim(-0.8, cursor - 0.1)
    ax.invert_yaxis()
    ax.axvspan(90.0, 100.5, color="#2ca02c", alpha=0.045, zorder=0)
    ax.axvline(90.0, color="black", linestyle="--", linewidth=1.8, zorder=1)
    ax.text(90.15, -0.42, "90% criterion", fontsize=14, ha="left", va="center")

    ylabels = []
    last_case = None
    for y, (kind, case_name, element, rec, samples) in zip(y_positions, rows):
        if kind == "element":
            color = COLORS[element]
            point = float(rec["point_estimate_percent"])
            lo = float(rec["conditional_fixed_RS_95CI_lower_percent"])
            hi = float(rec["conditional_fixed_RS_95CI_upper_percent"])
            mean = float(rec["conditional_fixed_RS_bootstrap_mean_percent"])
            label = f"  {element}"
            if case_name != last_case:
                ax.text(69.6, y - 0.47, CASE_LABELS[case_name], fontsize=16,
                        fontweight="bold", ha="right", va="bottom", clip_on=False)
                last_case = case_name
        elif kind == "case":
            color = "#333333"
            point = float(rec["point_macro_mean_percent"])
            lo = float(rec["conditional_fixed_RS_95CI_lower_percent"])
            hi = float(rec["conditional_fixed_RS_95CI_upper_percent"])
            mean = float(rec["conditional_fixed_RS_bootstrap_macro_mean_percent"])
            label = "  Site mean"
            ax.axhline(y + 0.5, color="#bdbdbd", linewidth=0.8)
        else:
            color = "#000000"
            point = float(rec["point_macro_mean_percent"])
            lo = float(rec["conditional_fixed_RS_95CI_lower_percent"])
            hi = float(rec["conditional_fixed_RS_95CI_upper_percent"])
            mean = float(rec["conditional_fixed_RS_bootstrap_macro_mean_percent"])
            label = "Overall mean (25 pairs)"
            ax.axhline(y - 0.55, color="black", linewidth=1.4)

        draw_ridge(ax, y, samples, color, height=0.30 if kind == "element" else 0.38,
                   alpha=0.28 if kind == "element" else 0.20)
        ax.hlines(y, lo, hi, color=color, linewidth=4.0 if kind != "element" else 3.0, zorder=4)
        ax.vlines([lo, hi], y - 0.11, y + 0.11, color=color, linewidth=1.4, zorder=4)
        ax.scatter(point, y, s=75 if kind == "element" else 115, color=color,
                   marker="o" if kind == "element" else "D", edgecolor="white", linewidth=0.8, zorder=5)
        ax.vlines(mean, y - 0.22, y + 0.22, color=color, linewidth=1.4, zorder=5)
        passed = lo >= 90.0
        ax.text(100.85, y, f"{point:5.2f}  [{lo:5.2f}, {hi:5.2f}]  {'✓' if passed else '×'}",
                fontsize=13.5, ha="left", va="center", color="#1b7837" if passed else "#b2182b",
                fontweight="bold" if kind != "element" else "normal", clip_on=False)
        ylabels.append(label)

    ax.set_yticks(y_positions, ylabels, fontsize=14)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[0] != "element":
            tick.set_fontweight("bold")
    ax.set_xticks(np.arange(70, 101, 5))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("CEMC probability mass contained in Random occupied-bin support (%)", fontsize=17)
    ax.set_ylabel("Facet / adsorption site and element", fontsize=17, labelpad=95)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Conditional slab-bootstrap containment of CEMC in Random support",
                 fontsize=21, fontweight="bold", y=0.975)
    ax.set_title(
        "Fixed empirical Random support; 20 slab runs, 10,000 bootstrap replicates; percentile 95% CI\n"
        "Ridge = bootstrap distribution, circle = full-data estimate, vertical tick = bootstrap mean, bar = 95% CI",
        fontsize=15, pad=24,
    )
    ax.text(100.85, -0.55, "Estimate  [95% CI]  pass", fontsize=13.5,
            ha="left", va="center", fontweight="bold", clip_on=False)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    OUTMETA.write_text(json.dumps({
        "figure": str(OUTPNG),
        "bootstrap": "conditional fixed empirical Random support",
        "bootstrap_unit": "slab run cluster",
        "n_slab_runs": 20,
        "n_bootstrap": 10000,
        "ci": "2.5th and 97.5th percentiles",
        "threshold_percent": 90.0,
        "rows": "25 case-element pairs, 5 case means, and 1 overall macro mean",
        "pdf_created": False,
    }, indent=2) + "\n")
    print(OUTPNG)


if __name__ == "__main__":
    main()
