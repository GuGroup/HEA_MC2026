# Figure 4: Surface-composition-weighted binding-energy density

a: PtPdRhRuIr fcc(111) OH top; b: FeCoNiPdPt fcc(100) H hollow. Columns: homogeneous, CEMC and CEMC with within-layer shuffling.

From the package root:

```bash
python Fig_4/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_4.png` (300 dpi) and `outputs/Fig_4.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

Zero BE shifts. All 19 stored CEMC temperatures are pooled, with 20 runs each; homogeneous distributions use the 20 initial slabs. Element curves integrate to the corresponding surface/zone-1 fractions. The original 13-bin Gaussian kernel (sigma=1.5 bins) is retained. Activity values are full-precision in metadata; plot labels show two significant figures. The figure values are reproduced from structures, not digitized from a raster. The manuscript prose mentions a layer-shuffled activity of 0.038 for Fe fcc(100); its figure and source calculations instead give 0.1621249547 (displayed 0.16).

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
