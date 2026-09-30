# Figure S14: H binding-free-energy densities for the remaining sites

Rows follow the supplied SI image: fcc(100) bridge, fcc(111) hollow, fcc(111) top, fcc(110) long-bridge. Columns: homogeneous, CEMC and CEMC with within-layer shuffling.

From the package root:

```bash
python Fig_S14/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_S14.png` (300 dpi) and `outputs/Fig_S14.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

Zero BE shifts. Original activity models and historical site-specific seeds are retained. All 19 CEMC temperatures and 20 runs are pooled. Densities use zone-1 element fractions; the optimal DeltaG_H is zero. Numeric data are recalculated from packed structures, including deterministic shuffled controls.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
