#!/usr/bin/env python3
"""Rebuild only the 1x5 probability-mass and paired-bootstrap figures."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


BASE = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
OUT = BASE / "oh_be_distribution_user_shift_20260824"
sys.path.insert(0, str(BASE))
import analyze_oh_be_user_shifts as analysis  # noqa: E402


SHIFTS = {
    "Ir": -0.04671,
    "Pd": -0.47713,
    "Pt": 0.02656,
    "Rh": 0.21891,
    "Ru": -0.24154,
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # The imported bootstrap seed is derived from this module-level mapping.
    analysis.SHIFTS = SHIFTS.copy()
    model = analysis.common.parse_model(analysis.common.MODEL_PATH)
    surface, _ = analysis.common.surface_sites_and_validate(model)

    random_by_run = []
    cemc_by_run = []
    for run in range(analysis.N_RUNS):
        slab_dir = analysis.SLABS / f"slab_{run + 1:02d}"
        random_occ = analysis.occupancy(slab_dir / "initial_random.cif")
        random_by_run.append(
            analysis.common.evaluate(model, surface, random_occ, SHIFTS)
        )

        values = []
        labels = []
        for temperature in analysis.TEMPERATURES:
            cemc_occ = analysis.occupancy(analysis.cemc_path(run, temperature))
            v, label = analysis.common.evaluate(model, surface, cemc_occ, SHIFTS)
            values.append(v)
            labels.append(label)
        cemc_by_run.append((np.concatenate(values), np.concatenate(labels)))

    random_pack = (
        np.concatenate([pack[0] for pack in random_by_run]),
        np.concatenate([pack[1] for pack in random_by_run]),
    )
    cemc_pack = (
        np.concatenate([pack[0] for pack in cemc_by_run]),
        np.concatenate([pack[1] for pack in cemc_by_run]),
    )
    all_values = np.concatenate(
        (random_pack[0], cemc_pack[0], [analysis.OPTIMUM, *analysis.WINDOW])
    )
    edges = np.linspace(float(all_values.min()) - 0.06, float(all_values.max()) + 0.06, 91)

    histogram_output = OUT / "probability_mass_weighted_histograms_1x5.png"
    bootstrap_output = OUT / "paired_joint_bootstrap.png"
    containment, bin_rows = analysis.plot_1x5(
        random_pack, cemc_pack, edges, histogram_output
    )
    full, samples = analysis.bootstrap(random_by_run, cemc_by_run, edges)
    bootstrap_rows = analysis.plot_bootstrap(full, samples, bootstrap_output)

    write_csv(OUT / "probability_mass_weighted_histograms_1x5_bin_data_requested_shift.csv", bin_rows)
    write_csv(OUT / "paired_joint_bootstrap_summary_requested_shift.csv", bootstrap_rows)
    np.savez_compressed(
        OUT / "paired_joint_bootstrap_replicates_requested_shift.npz",
        elements=np.asarray(analysis.ELEMENTS),
        containment=samples,
        element_mean=samples.mean(axis=1),
        full_data=full,
        bin_edges_eV=edges,
    )
    metadata = {
        "formula": "E_OH = intercept + zone1(top) + sum(zone2) + sum(zone3) - BE_shift(top element)",
        "BE_shifts_eV": SHIFTS,
        "elements": list(analysis.ELEMENTS),
        "temperatures_K": list(analysis.TEMPERATURES),
        "n_random_slabs": analysis.N_RUNS,
        "n_cemc_structures": analysis.N_RUNS * len(analysis.TEMPERATURES),
        "OH_sites_per_structure": int(len(surface)),
        "histogram_bins": len(edges) - 1,
        "histogram_bin_edges_eV": edges.tolist(),
        "containment_full_data": containment,
        "bootstrap": {
            "n_replicates": analysis.N_BOOT,
            "unit": "slab run",
            "paired_random_cemc": True,
            "cemc_temperatures_retained_per_run": len(analysis.TEMPERATURES),
            "rows": bootstrap_rows,
        },
        "outputs": {
            "probability_mass_weighted_histograms_1x5": str(histogram_output),
            "paired_joint_bootstrap": str(bootstrap_output),
        },
    }
    metadata_path = OUT / "probability_mass_and_paired_bootstrap_requested_shift_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    validation = {
        "random_site_count": int(len(random_pack[0])),
        "cemc_site_count": int(len(cemc_pack[0])),
        "bootstrap_shape": list(samples.shape),
        "all_finite": bool(np.isfinite(samples).all()),
        "mass_sums_one": all(
            abs(sum(row["random_probability_mass"] for row in bin_rows if row["element"] == element) - 1.0) < 1e-12
            and abs(sum(row["cemc_probability_mass"] for row in bin_rows if row["element"] == element) - 1.0) < 1e-12
            for element in analysis.ELEMENTS
        ),
    }
    print(json.dumps({"shifts": SHIFTS, "containment": containment, "bootstrap": bootstrap_rows, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
