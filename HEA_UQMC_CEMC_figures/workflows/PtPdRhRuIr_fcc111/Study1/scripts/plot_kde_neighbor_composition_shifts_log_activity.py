#!/usr/bin/env python3
"""Plot composition-shift histograms for log-activity KDE-neighbor trials."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neighbors", type=Path, default=Path("results/kde_neighbor_be_temperature_log/log_activity_kde_peak_nearest_trials.csv"))
    p.add_argument("--exporter", type=Path, default=Path("scripts/export_neighbor_composition_shifts_log_activity.cpp"))
    p.add_argument("--output-dir", type=Path, default=Path("results/kde_neighbor_composition_shifts_log"))
    p.add_argument("--n-compositions", type=int, default=1400)
    p.add_argument("--random-seed", type=int, default=20260706)
    p.add_argument("--comp-mean", type=float, default=0.000347)
    p.add_argument("--comp-sigma", type=float, default=0.046858)
    p.add_argument("--bin-width", type=float, default=0.01)
    p.add_argument("--dpi", type=int, default=240)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighbors = pd.read_csv(args.neighbors).sort_values("neighbor_rank")
    trial_list = args.output_dir / "neighbor_trials.csv"
    neighbors[["trial", "neighbor_rank"]].to_csv(trial_list, index=False)
    executable = args.output_dir / "export_neighbor_composition_shifts_log_activity"
    shifts_csv = args.output_dir / "log_activity_neighbor_composition_shifts.csv"
    subprocess.run(["g++", "-O2", "-std=c++11", str(args.exporter), "-o", str(executable)], check=True)
    subprocess.run([
        str(executable), str(trial_list), str(args.n_compositions), str(args.random_seed),
        str(args.comp_mean), str(args.comp_sigma), str(shifts_csv),
    ], check=True)
    shifts = pd.read_csv(shifts_csv).merge(
        neighbors[["trial", "neighbor_rank"]], on="trial", how="left", validate="many_to_one"
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
    global_max = 0
    plot_data = []
    representative_rows = []
    for element in ELEMENTS:
        column = f"comp_shift_{element}"
        values = shifts[column].to_numpy(float)
        left = np.floor(values.min() / args.bin_width) * args.bin_width
        right = np.ceil(values.max() / args.bin_width) * args.bin_width
        if right <= left: right = left + args.bin_width
        edges = np.arange(left, right + args.bin_width * 1.01, args.bin_width)
        counts, _ = np.histogram(values, bins=edges)
        modal = int(np.argmax(counts))
        in_bin = (values >= edges[modal]) & (values < edges[modal + 1])
        if modal == len(counts) - 1: in_bin |= values == edges[modal + 1]
        candidates = shifts.loc[in_bin].sort_values(["neighbor_rank", "composition_index"], kind="stable")
        representative = candidates.iloc[0]
        representative_rows.append({
            "element": element, "trial": int(representative.trial),
            "composition_index": int(representative.composition_index),
            "composition_shift": float(representative[column]),
            "modal_bin_left": edges[modal], "modal_bin_right": edges[modal + 1],
            "modal_bin_count": int(counts[modal]), "n_samples": len(values),
        })
        global_max = max(global_max, int(counts.max()))
        plot_data.append((column, edges, counts, modal, representative))

    for ax, element, (_, edges, counts, modal, representative) in zip(axes, ELEMENTS, plot_data):
        colors = ["#5375C6"] * len(counts); colors[modal] = "#D98245"
        ax.bar(edges[:-1], counts, width=args.bin_width * 0.92, align="edge", color=colors,
               edgecolor="white", linewidth=0.4)
        value = float(representative[f"comp_shift_{element}"])
        ax.axvline(value, color="#C9342B", linestyle="--", linewidth=1.8)
        ax.set_title(
            f"{element}: {value:.4f}\n"
            f"trial {int(representative.trial)}, comp {int(representative.composition_index)}\n"
            f"bin count {counts[modal]}/{len(shifts)}"
        )
        ax.set_xlabel("Composition shift"); ax.set_ylim(0, max(1, global_max * 1.08)); ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(f"Count among {len(neighbors)} trials x {args.n_compositions} compositions")
    peak_trial = int(neighbors.iloc[0].trial)
    fig.suptitle(
        f"Log-activity tau>0 KDE-peak nearest {len(neighbors)} trials: element-wise composition shifts\n"
        f"peak trial {peak_trial}; fixed {args.bin_width:.2f} bins; orange = modal bin; red dashed = selected actual value",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = args.output_dir / "log_activity_top100_composition_shift_histograms.png"
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(representative_rows).to_csv(args.output_dir / "log_activity_modal_bin_representative_composition_shifts.csv", index=False)
    print(f"Wrote {output}")
    print(f"Wrote {shifts_csv}")
    print(f"Wrote {args.output_dir / 'log_activity_modal_bin_representative_composition_shifts.csv'}")

if __name__ == "__main__":
    main()
