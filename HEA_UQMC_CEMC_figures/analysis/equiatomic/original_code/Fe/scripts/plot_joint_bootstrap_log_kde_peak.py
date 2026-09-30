#!/usr/bin/env python3
"""Separate paired-joint slab-bootstrap ridge/forest plot for each facet/site."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import analyze_hbe_distributions_log_kde_peak as analysis
import bootstrap_probability_mass_log_kde_peak as bootstrap


OUTDIR = bootstrap.OUTPUT_ROOT / "by_facet_site"
XMIN, XMAX = 45.0, 100.5
COLORS = dict(zip(analysis.ELEMENTS, ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]))
CASE_LABELS = {case: f"FeCoNiPdPt fcc({facet}) {site}"
               for facet, site, case in analysis.COMBOS}


def draw_ridge(ax, y0, samples, color, height=0.34, alpha=0.28):
    lo, hi = np.percentile(samples, [0.1, 99.9])
    pad = max(0.18, 0.35 * (hi - lo))
    x = np.linspace(max(XMIN, lo - pad), min(XMAX, hi + pad), 220)
    d = gaussian_kde(samples)(x)
    d /= d.max()
    ax.fill_between(x, y0, y0 + height * d, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y0 + height * d, color=color, linewidth=1.2)


def plot_case(case_name, pair, case_row, pair_rep, case_rep):
    subset = pair[pair["case"] == case_name].reset_index(drop=True)
    pair_indices = np.flatnonzero(pair["case"].to_numpy() == case_name)
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    fig.subplots_adjust(left=0.13, right=0.73, top=0.80, bottom=0.16)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-0.65, 5.7)
    ax.invert_yaxis()
    ax.axvspan(90.0, XMAX, color="#2ca02c", alpha=0.045, zorder=0)
    ax.axvline(90.0, color="black", linestyle="--", linewidth=2.0, zorder=1)
    ax.text(90.3, -0.38, "90% criterion", fontsize=16, ha="left", va="center")

    for y, element in enumerate(analysis.ELEMENTS):
        rec = subset.iloc[y]
        samples = pair_rep[pair_indices[y]]
        color = COLORS[element]
        point = float(rec["point_estimate_percent"])
        mean = float(rec["paired_joint_bootstrap_mean_percent"])
        lo = float(rec["paired_joint_95CI_lower_percent"])
        hi = float(rec["paired_joint_95CI_upper_percent"])
        draw_ridge(ax, y, samples, color)
        ax.hlines(y, lo, hi, color=color, linewidth=3.4, zorder=4)
        ax.vlines([lo, hi], y - 0.13, y + 0.13, color=color, linewidth=1.6, zorder=4)
        ax.scatter(point, y, s=100, color=color, edgecolor="white", linewidth=0.9, zorder=5)
        ax.vlines(mean, y - 0.24, y + 0.24, color=color, linewidth=1.7, zorder=5)
        passed = lo >= 90.0
        ax.text(100.85, y, f"{point:5.2f}  [{lo:5.2f}, {hi:5.2f}]  {'✓' if passed else '×'}",
                fontsize=16, ha="left", va="center",
                color="#1b7837" if passed else "#b2182b", clip_on=False)

    y = 5.25
    point = float(case_row["point_macro_mean_percent"])
    mean = float(case_row["paired_joint_bootstrap_macro_mean_percent"])
    lo = float(case_row["paired_joint_95CI_lower_percent"])
    hi = float(case_row["paired_joint_95CI_upper_percent"])
    draw_ridge(ax, y, case_rep, "#333333", height=0.40, alpha=0.20)
    ax.hlines(y, lo, hi, color="#333333", linewidth=4.4, zorder=4)
    ax.vlines([lo, hi], y - 0.14, y + 0.14, color="#333333", linewidth=1.7, zorder=4)
    ax.scatter(point, y, s=145, color="#333333", marker="D", edgecolor="white", linewidth=0.9, zorder=5)
    ax.vlines(mean, y - 0.27, y + 0.27, color="#333333", linewidth=1.8, zorder=5)
    passed = lo >= 90.0
    ax.text(100.85, y, f"{point:5.2f}  [{lo:5.2f}, {hi:5.2f}]  {'✓' if passed else '×'}",
            fontsize=16, fontweight="bold", ha="left", va="center",
            color="#1b7837" if passed else "#b2182b", clip_on=False)
    ax.axhline(4.65, color="black", linewidth=1.2)

    ax.set_yticks([0, 1, 2, 3, 4, y], [*analysis.ELEMENTS, "Site mean"], fontsize=17)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xticks(np.arange(45, 101, 5))
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("CEMC probability mass contained in resampled Random occupied-bin support (%)", fontsize=18)
    ax.set_ylabel("Element", fontsize=18)
    ax.text(100.85, -0.38, "Estimate  [95% CI]  pass", fontsize=16,
            ha="left", va="center", fontweight="bold", clip_on=False)
    fig.suptitle(
        f"{CASE_LABELS[case_name]}: paired joint slab-bootstrap containment",
        fontsize=22, fontweight="bold", y=0.965,
    )
    ax.set_title(
        "Random support and CEMC mass jointly resampled by paired slab run; 20 runs, 10,000 replicates; percentile 95% CI\n"
        "Ridge = bootstrap distribution, circle = full-data estimate, vertical tick = bootstrap mean, bar = 95% CI",
        fontsize=15.2, pad=22,
    )
    outfile = OUTDIR / f"{case_name}_joint_bootstrap_ridgeline_forest.png"
    fig.savefig(outfile, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outfile


def main():
    global XMIN
    pair = pd.read_csv(bootstrap.OUTPUT_PAIR)
    case = pd.read_csv(bootstrap.OUTPUT_CASE)
    reps = np.load(bootstrap.OUTPUT_NPZ)
    pair_rep = reps["pair_joint_percent"]
    case_rep = reps["case_joint_percent"]
    XMIN = max(0.0, float(np.floor(np.percentile(pair_rep, 0.05) / 5.0) * 5.0 - 2.5))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for case_idx, case_name in enumerate(case["case"]):
        outputs.append(str(plot_case(case_name, pair, case.iloc[case_idx],
                                     pair_rep, case_rep[case_idx])))
    (OUTDIR / "metadata.json").write_text(json.dumps({
        "figures": outputs,
        "bootstrap": "paired joint Random-support and CEMC-mass slab-run cluster bootstrap",
        "n_slab_runs": 20,
        "n_bootstrap": 10000,
        "ci": "2.5th and 97.5th percentiles",
        "common_x_range_percent": [XMIN, XMAX],
        "pdf_created": False,
    }, indent=2) + "\n")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
