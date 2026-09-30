# Figure S13: Final layer-pair elemental composition

a: PtPdRhRuIr fcc(111); b: FeCoNiPdPt fcc(100) hollow run; c: FeCoNiPdPt fcc(110) bridge run; d: FeCoNiPdPt fcc(111) hollow run.

From the package root:

```bash
python Fig_S13/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_S13.png` (300 dpi) and `outputs/Fig_S13.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

Final 298 K snapshots, 20 slabs per panel. Symmetric top/bottom layer pairs are pooled (surface: layers 1+10, subsurface: 2+9, etc.). Every row averages 4,000 atomic sites. Original full-precision percentages are supplied and verified directly from packed atomic numbers. No values transcribed from a figure are used.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
