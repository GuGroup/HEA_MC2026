#!/usr/bin/env python3
"""Plot 1x3 deltaG_H distributions using the most-probable 3D-histogram trial."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_hbe_distributions_no_shift_1x3_all as analysis


OUTPUT_SUBDIR = "most_probable_trial_be_shift_1x3"
DISPLAY_TITLES = {
    "Random": "Homogeneous",
    "CEMC": "CEMC",
    "CEMC + layer shuffle": "CEMC + layer shuffle",
}
ACTIVITY_POSITIONS = {
    "fcc100_bridge": {
        "Random": (0.96, 0.68, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc100_hollow": {
        "Random": (0.04, 0.68, "left"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc110_bridge": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.96, 0.95, "right"),
        "CEMC + layer shuffle": (0.96, 0.95, "right"),
    },
    "fcc111_hollow": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc111_top": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.linewidth": 1.2,
    }
)


def pack_evaluated(model, evaluated):
    packed = {}
    for method in analysis.METHODS:
        final_be = np.concatenate([row[2] for row in evaluated[method]])
        packed[method] = (
            final_be - model.e_opt,
            np.vstack([row[3] for row in evaluated[method]]),
        )
    return packed


def draw_figure(packed, model, case: str, output_png: Path) -> None:
    activities = {
        method: analysis.activity_no_log(packed[method][0], model.activity_temperature)
        for method in analysis.METHODS
    }
    combined = np.concatenate([packed[method][0] for method in analysis.METHODS])
    lo = min(float(combined.min()), analysis.OPTIMAL_DELTA_G_H_EV) - 0.06
    hi = max(float(combined.max()), analysis.OPTIMAL_DELTA_G_H_EV) + 0.06
    edges = np.linspace(lo, hi, 91)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(18.9, 6.4), sharex=True, sharey=True)
    for method, ax in zip(analysis.METHODS, axes):
        values, fractions = packed[method]
        for element_index, element in enumerate(analysis.ELEMENTS):
            density = analysis.smoothed_weighted_density(
                values, fractions[:, element_index], edges, len(values)
            )
            ax.plot(
                centers,
                density,
                lw=2.7,
                color=analysis.ELEMENT_COLORS[element],
                label=element,
            )
        ax.axvline(
            analysis.OPTIMAL_DELTA_G_H_EV,
            ls="--",
            lw=2.1,
            color="black",
            label=r"Optimal $\Delta G_{\mathrm{H}}$",
        )
        ax.set_title(DISPLAY_TITLES[method], pad=10)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.tick_params(width=1.2, length=5)
        ax.grid(False)
        activity_x, activity_y, alignment = ACTIVITY_POSITIONS[case][method]
        ax.text(
            activity_x,
            activity_y,
            f"Activity = {activities[method]:.6f}",
            transform=ax.transAxes,
            ha=alignment,
            va="top",
            fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3.0),
            zorder=90,
        )

    axes[0].set_ylabel("Probability density")
    handles, labels = axes[0].get_legend_handles_labels()
    kept = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if not label.startswith("Optimal")
    ]
    handles, labels = zip(*kept)
    legend = axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        columnspacing=1.4,
        handlelength=2.4,
        fontsize=18,
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_case(facet: str, site: str, case: str) -> Path:
    workdir = analysis.BASE / facet / site
    plots_root = (
        analysis.OUTPUT_ROOT
        / "by_facet_site"
        / case
        / analysis.ANALYSIS_1X3
        / "plots_by_temperature"
    )
    output_dir = plots_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model = analysis.parse_model(workdir / "inputs/activity_model.txt")
    groups = analysis.layer_groups(workdir / "inputs/template.cif")
    be_df, be_shifts, _ = analysis.load_shift_data(case)
    shift_source = (
        analysis.SHIFT_ROOT
        / case
        / "kde_smoothed"
        / "log_kde_representative_trial_BE_shifts.csv"
    )
    source_df = analysis.pd.read_csv(shift_source)
    selected_trial = int(source_df.iloc[0]["trial"])
    applied = be_df.reset_index().rename(columns={"index": "element"})
    applied.insert(0, "case", case)
    applied.insert(1, "representative_selection", source_df.iloc[0]["representative_selection"])
    applied.insert(2, "selection_trial", selected_trial)
    applied["source_csv"] = str(shift_source)
    applied.to_csv(output_dir / "applied_most_probable_trial_be_shifts.csv", index=False)

    pooled_values = {method: [] for method in analysis.METHODS}
    pooled_fractions = {method: [] for method in analysis.METHODS}
    with tempfile.TemporaryDirectory(prefix=f"most_probable_shift_{case}_") as tmp:
        tmpdir = Path(tmp)
        random_eval = [
            analysis.evaluate_sites(
                model, analysis.reconstruct_random(workdir, run, tmpdir), be_shifts
            )
            for run in range(20)
        ]
        for temperature in analysis.TEMPERATURES:
            cemc = analysis.load_cemc_records(workdir, temperature)
            evaluated = {
                "Random": random_eval,
                "CEMC": [],
                "CEMC + layer shuffle": [],
            }
            for run in range(20):
                occ = cemc[run]
                evaluated["CEMC"].append(
                    analysis.evaluate_sites(model, occ, be_shifts)
                )
                shuffled = analysis.shuffle_occ(
                    occ,
                    groups,
                    analysis.stable_seed(
                        "FeCoNiPdPt-equi-layer-shuffle-v1",
                        f"{facet}/{site}",
                        run,
                        temperature,
                    ),
                )
                evaluated["CEMC + layer shuffle"].append(
                    analysis.evaluate_sites(model, shuffled, be_shifts)
                )

            packed = pack_evaluated(model, evaluated)
            stem = "final_0298K" if temperature == 298 else f"T_{temperature:04d}K"
            output_png = output_dir / (
                f"{stem}_element_deltaG_H_distribution_"
                "most_probable_trial_be_shift_1x3.png"
            )
            draw_figure(packed, model, case, output_png)
            for method in analysis.METHODS:
                pooled_values[method].append(packed[method][0])
                pooled_fractions[method].append(packed[method][1])
            print(f"DONE {case} {temperature}K", flush=True)

    packed_all = {
        method: (
            np.concatenate(pooled_values[method]),
            np.vstack(pooled_fractions[method]),
        )
        for method in analysis.METHODS
    }
    draw_figure(
        packed_all,
        model,
        case,
        output_dir
        / "all_temperatures_average_element_deltaG_H_distribution_"
        "most_probable_trial_be_shift_1x3.png",
    )
    (output_dir / "README.txt").write_text(
        "Selection: maximum_3D_histogram_KDE_probability_CEMC\n"
        f"Selected trial: {selected_trial}\n"
        "Applied values: exact per-element BE shifts from applied_most_probable_trial_be_shifts.csv\n"
        "Methods: Homogeneous, CEMC, CEMC + layer shuffle\n"
        "Temperatures: 298 K and 300-2000 K; 20 runs per temperature\n",
        encoding="utf-8",
    )
    print(f"OUTPUT {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[case for _, _, case in analysis.COMBOS])
    args = parser.parse_args()
    for facet, site, case in analysis.COMBOS:
        if args.case is not None and case != args.case:
            continue
        process_case(facet, site, case)


if __name__ == "__main__":
    main()
