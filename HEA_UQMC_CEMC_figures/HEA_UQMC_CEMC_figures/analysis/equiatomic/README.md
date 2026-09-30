# Equiatomic figure analysis

Figures 3-6 and S13-S15 can be plotted immediately from their numeric data. Run `python reproduce_all_figures.py` at the package root to plot all 16 packaged figures, or `python Fig_4/plot.py` for one figure. No CEMC, server access or machine-specific paths are needed for this step.

## Recompute and verify the numerical inputs

```bash
python analysis/equiatomic/recompute.py
```

This reads all six historical case ensembles (four physical facets), reconstructs shuffled controls with the original seeds, evaluates the original adsorption models, regenerates Fig. 4/S14 densities and cached site predictions, and compares the results with original Fig. 3, 5, 6, S13 and S15 tables. It writes `validation/equiatomic_numeric_checks.json`. The largest difference in the initial verification was below 5e-14. Run the individual figure plot scripts again after recomputing.

The original density smoothing is a 13-bin Gaussian kernel with sigma=1.5 bins. Temperature-curve smoothing is a separate Gaussian temperature kernel, sigma=50 K and cutoff=150 K. Fig. 3 uses the top surface, while S13 averages symmetric layer pairs. Fig. 4/S14 pool 19 snapshots per CEMC run, not only the final 298 K state. Activity is the arithmetic mean of per-site volcano activities.

`evaluate.py` is the portable evaluator used by each facet's `equi-atomic/run.py`. `pt_core.py`, `fe_core.py` and `surface_core.py` retain the original scientific functions; the wrapper calls those functions with packaged inputs. Their original standalone main functions use historical absolute paths and are not the entry points for this package. `original_code/` preserves the source analysis and plotting scripts for audit, including exploratory and historical versions; use the documented portable entry points above to reproduce the selected paper figures.

Fig. 6/S15 use the historical nonzero BE shifts and conditional probability normalization. Fig. 4/5/S14 use zero shifts. All shifts are explicit in case/figure metadata. Paper Fig. 5 deliberately retains its two different original normalization conventions (Pt conditional mass versus Fe weighted density). Source provenance and SHA256 hashes are in `validation/equiatomic_provenance.json` and the root manifest.

The original figure panels are retained as references. Regenerated composite figures have the same data and panel meaning, with portable Matplotlib typography and spacing. They are not asserted to be pixel-identical to Word's assembled images.
