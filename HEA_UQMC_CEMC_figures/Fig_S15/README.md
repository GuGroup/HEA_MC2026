# Figure S15: Element-conditioned H probability mass for the remaining sites

Rows: fcc(100) bridge, fcc(110) long-bridge, fcc(111) hollow, fcc(111) top.

From the package root:

```bash
python Fig_S15/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_S15.png` (300 dpi) and `outputs/Fig_S15.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

The exact nonzero per-element BE shifts used by the source plots are supplied in each case folder. Original weighted conditional masses, homogeneous occupied-bin support and CEMC containment are preserved. These shifts differ from the zero-shift Fig. S14; do not mix their energy coordinates.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
