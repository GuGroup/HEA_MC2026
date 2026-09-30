# Figure 6: Element-conditioned probability mass and containment

a: PtPdRhRuIr fcc(111) OH top; b: FeCoNiPdPt fcc(100) H hollow.

From the package root:

```bash
python Fig_6/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_6.png` (300 dpi) and `outputs/Fig_6.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

This source figure uses the supplied nonzero BE shifts, unlike Fig. 4 and Fig. 5. Exact shifts and source selection metadata are included. Conditional probability masses normalize separately for each element; Fe multiatom adsorption sites use their fractional zone-1 element weights. Containment is the CEMC mass in bins occupied by homogeneous sites, not merely a min-max interval. The blue/red partition and containment percentages are calculated from the supplied bin tables.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
