# Figure 3: Temperature-dependent surface composition and Warren-Cowley order

a: PtPdRhRuIr fcc(111) composition; b: FeCoNiPdPt fcc(100) hollow-run composition; c/d: their surface 1NN Warren-Cowley parameters.

From the package root:

```bash
python Fig_3/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_3.png` (300 dpi) and `outputs/Fig_3.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

20 runs at each of 172 temperatures. The topmost layer is used (not both surfaces). Gaussian smoothing uses actual temperature separations, sigma=50 K and a 3-sigma cutoff. Missing WC pairs remain excluded from the mean; their valid-run counts are retained. Raw CSV values are supplied alongside smoothed values.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
