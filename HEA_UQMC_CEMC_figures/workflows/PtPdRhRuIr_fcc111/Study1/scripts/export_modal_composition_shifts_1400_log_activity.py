#!/usr/bin/env python3
"""Select per-composition modal-bin shifts from log-activity KDE neighbors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")


def select_modal(values: pd.DataFrame, column: str, bin_width: float) -> tuple[float, int, float]:
    shifts = values[column].to_numpy(float)
    # Integer bin IDs make globally fixed bins anchored at integer multiples of bin_width.
    bin_ids = np.floor(shifts / bin_width).astype(np.int64)
    unique_ids, counts = np.unique(bin_ids, return_counts=True)
    modal_id = unique_ids[np.flatnonzero(counts == counts.max())[0]]
    candidates = values.loc[bin_ids == modal_id].sort_values("neighbor_rank", kind="stable")
    representative = candidates.iloc[0]
    return float(representative[column]), int(representative["trial"]), float(counts.max() / len(values))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shifts", type=Path, default=Path("results/kde_neighbor_composition_shifts_log/log_activity_neighbor_composition_shifts.csv"))
    p.add_argument("--neighbors", type=Path, default=Path("results/kde_neighbor_be_temperature_log/log_activity_kde_peak_nearest_trials.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/kde_neighbor_composition_shifts_log"))
    p.add_argument("--n-compositions", type=int, default=1400)
    p.add_argument("--bin-width", type=float, default=0.01)
    args = p.parse_args()

    shifts = pd.read_csv(args.shifts)
    neighbors = pd.read_csv(args.neighbors)[["trial", "neighbor_rank"]]
    if "neighbor_rank" not in shifts:
        shifts = shifts.merge(neighbors, on="trial", how="left", validate="many_to_one")
    expected = np.arange(args.n_compositions)
    actual = np.sort(shifts["composition_index"].unique())
    if not np.array_equal(actual, expected):
        raise ValueError(f"Expected composition indices 0..{args.n_compositions - 1}")

    selected_values = np.empty((len(ELEMENTS), args.n_compositions), dtype=float)
    selected_trials = np.empty((len(ELEMENTS), args.n_compositions), dtype=int)
    modal_probabilities = np.empty((len(ELEMENTS), args.n_compositions), dtype=float)
    grouped = {int(comp): part for comp, part in shifts.groupby("composition_index", sort=False)}
    for comp in expected:
        part = grouped[int(comp)]
        for row, element in enumerate(ELEMENTS):
            value, trial, probability = select_modal(part, f"comp_shift_{element}", args.bin_width)
            selected_values[row, comp] = value
            selected_trials[row, comp] = trial
            modal_probabilities[row, comp] = probability

    args.output_dir.mkdir(parents=True, exist_ok=True)
    value_path = args.output_dir / "log_activity_modal_composition_shifts_1400_columns.csv"
    trial_path = args.output_dir / "log_activity_modal_composition_shift_trials_1400_columns.csv"
    probability_path = args.output_dir / "log_activity_modal_composition_shift_probabilities_1400_columns.csv"

    def write_transposed(values: np.ndarray, path: Path) -> None:
        table = pd.DataFrame(values.T, columns=ELEMENTS)
        table.insert(0, "composition_index", expected)
        table.to_csv(path, index=False)

    write_transposed(selected_values, value_path)
    write_transposed(selected_trials, trial_path)
    write_transposed(modal_probabilities, probability_path)
    n_columns = len(ELEMENTS) + 1
    print(f"Wrote {value_path} ({args.n_compositions} rows x {n_columns} columns)")
    print(f"Wrote {trial_path} ({args.n_compositions} rows x {n_columns} columns)")
    print(f"Wrote {probability_path} ({args.n_compositions} rows x {n_columns} columns)")

if __name__ == "__main__":
    main()
