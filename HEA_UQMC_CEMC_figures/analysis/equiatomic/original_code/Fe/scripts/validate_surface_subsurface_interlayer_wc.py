import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
CASES = [("100", "bridge"), ("100", "hollow"), ("110", "bridge"),
         ("111", "hollow"), ("111", "top")]
METHODS = ["Random", "CEMC", "CEMC + layer shuffle"]
ELEMENTS = ["Fe", "Co", "Ni", "Pd", "Pt"]


def main():
    checks = []
    for facet, site in CASES:
        directory = BASE / facet / site / "surface_subsurface_interlayer_wc_analysis"
        csv_path = directory / "all_temperatures_surface_subsurface_interlayer_1NN_WC.csv"
        png_path = directory / "all_temperatures_surface_subsurface_interlayer_1NN_WC_heatmap.png"
        metadata = json.loads((directory / "metadata.json").read_text())
        table = pd.read_csv(csv_path)
        assert len(table) == 75
        assert png_path.exists() and png_path.stat().st_size > 10_000
        assert not list(directory.glob("*.pdf"))
        assert metadata["surface_layer_atoms"] == 100
        assert metadata["subsurface_layer_atoms"] == 100
        assert metadata["random_structures"] == 20
        assert metadata["cemc_structures"] == 380
        assert metadata["layer_shuffled_structures"] == 380
        assert "both directions" in metadata["matrix_direction"]
        method_checks = {}
        for method in METHODS:
            subset = table[table.method == method]
            matrix = subset.pivot(index="center_element", columns="neighbor_element",
                                  values="mean_WC_alpha").reindex(index=ELEMENTS, columns=ELEMENTS).to_numpy()
            assert np.isfinite(matrix).all()
            symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
            assert symmetry_error < 1.0e-12, symmetry_error
            assert float(np.max(matrix)) <= 1.0 + 1.0e-12
            expected_n = 20 if method == "Random" else 380
            assert set(subset.n_input_structures) == {expected_n}
            method_checks[method] = {
                "max_symmetry_error": symmetry_error,
                "min_WC": float(np.min(matrix)),
                "max_WC": float(np.max(matrix)),
            }
        checks.append({
            "facet": facet, "site": site,
            "coordination": metadata["surface_to_subsurface_1NN_coordination"],
            "distance_A": metadata["surface_to_subsurface_1NN_distance_A"],
            "methods": method_checks,
        })

    combined = pd.read_csv(BASE / "surface_subsurface_interlayer_WC_all_facets_sites.csv")
    assert len(combined) == 375
    result = {
        "validated": True,
        "definition": "bidirectional surface-subsurface interlayer WC, directionally normalized and averaged",
        "n_cases": 5,
        "n_combined_rows": 375,
        "n_png": 5,
        "n_pdf": 0,
        "cases": checks,
    }
    output = BASE / "surface_subsurface_interlayer_WC_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
