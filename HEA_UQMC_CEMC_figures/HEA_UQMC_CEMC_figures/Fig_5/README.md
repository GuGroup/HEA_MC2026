# Figure 5: Homogeneous binding-energy distributions and activity volcano

a: PtPdRhRuIr fcc(111) OH top; b: FeCoNiPdPt fcc(100) H hollow.

From the package root:

```bash
python Fig_5/plot.py
```

Alternatively run `python plot.py` in this folder. The script resolves data relative to its own location and needs only this figure folder and Python dependencies from the root `requirements.txt`. Outputs are `outputs/Fig_5.png` (300 dpi) and `outputs/Fig_5.pdf` (vector). Supplied reference PNGs are provenance/comparison files, never inputs to `plot.py`.

Zero BE shifts; 20 homogeneous slabs. Pt uses conditional per-element probability mass, whereas the original Fe panel uses zone-1-fraction-weighted probability density. These original, different normalizations are preserved and documented in the per-system metadata. Site-level energies and histogram tables are supplied. The volcano is exp(-abs(E-E_opt)/(k_B*298 K)), with Pt E_opt=1.1 eV and Fe DeltaG_opt=0 eV.

`data/` contains the exact source tables or newly recalculated numeric curves with their metadata. Plot layout is assembled by the portable script; fonts and panel spacing may differ from the embedded Word figure. Scientific values, binning and statistical conventions are preserved. To repeat the structure-to-energy/surface analysis and numeric comparisons, run `python analysis/equiatomic/recompute.py` from the package root. To rerun CEMC itself, use the appropriate `workflows/FACET/equi-atomic/README.md`.
