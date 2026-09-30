#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
COMBOS = (("100", "bridge"), ("100", "hollow"), ("110", "bridge"),
          ("111", "hollow"), ("111", "top"))
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
METHODS = ("Random", "CEMC", "CEMC + layer shuffle")
EXPECTED_COORDINATION = {"100": 4, "110": 2, "111": 6}


result = {"validated": False, "cases": {}}
for facet, site in COMBOS:
    case = f"fcc{facet}_{site}"
    root = BASE / facet / site / "surface_inplane_wc_analysis"
    pngs = list(root.glob("*.png"))
    pdfs = list(root.glob("*.pdf"))
    assert len(pngs) == 1 and len(pdfs) == 0
    data = pd.read_csv(root / "all_temperatures_surface_layer_inplane_1NN_WC.csv")
    assert len(data) == 75
    assert set(data["method"]) == set(METHODS)
    assert set(data["center_element"]) == set(ELEMENTS)
    assert set(data["neighbor_element"]) == set(ELEMENTS)
    assert data["mean_WC_alpha"].notna().all()
    assert float(data["mean_WC_alpha"].max()) <= 1.0 + 1e-12
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["surface_layer_atoms"] == 100
    assert metadata["surface_inplane_1NN_coordination"] == EXPECTED_COORDINATION[facet]
    assert metadata["random_structures"] == 20
    assert metadata["cemc_structures"] == 380
    assert metadata["layer_shuffled_structures"] == 380

    method_stats = {}
    matrices = {}
    for method in METHODS:
        part = data[data["method"] == method]
        matrix = part.pivot(index="center_element", columns="neighbor_element",
                            values="mean_WC_alpha").loc[list(ELEMENTS), list(ELEMENTS)].to_numpy(float)
        symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
        assert symmetry_error < 1e-12
        matrices[method] = matrix
        method_stats[method] = {
            "max_abs_mean_WC": float(np.max(np.abs(matrix))),
            "matrix_symmetry_error": symmetry_error,
            "n_input_structures": int(part["n_input_structures"].iloc[0]),
        }
    delta = matrices["CEMC"] - matrices["CEMC + layer shuffle"]
    strongest = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    result["cases"][case] = {
        "png_files": 1, "pdf_files": 0, "csv_rows": len(data),
        "surface_inplane_1NN_coordination": metadata["surface_inplane_1NN_coordination"],
        "surface_inplane_1NN_distance_A": metadata["surface_inplane_1NN_distance_A"],
        "method_stats": method_stats,
        "strongest_CEMC_minus_shuffle_pair": f"{ELEMENTS[strongest[0]]}-{ELEMENTS[strongest[1]]}",
        "strongest_CEMC_minus_shuffle_delta_alpha": float(delta[strongest]),
    }

combined = pd.read_csv(BASE / "surface_layer_inplane_WC_all_facets_sites.csv")
assert len(combined) == 375
result.update({"total_heatmap_png": 5, "total_pdf": 0, "combined_csv_rows": len(combined),
               "validated": True})
(BASE / "surface_layer_inplane_WC_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
