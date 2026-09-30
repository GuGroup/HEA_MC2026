# FeCoNiPdPt_fcc110: equiatomic CEMC

This independent subfolder contains the historical equiatomic calculation, with original CE and adsorption-model exports, C++ sources, configurations and all 20 saved runs. It does not depend on the UQMC workflow in the parent folder. Available `--site` values: **bridge**.

Install the root `requirements.txt`. CEMC generation requires Linux, a C++17 compiler, and MPI (`mpicxx`, `mpirun`, `g++`). Figure plotting, exported structure reading and adsorption evaluation require only Python. Commands below are run in this folder.

## Reuse the stored structures

```bash
# Export all 20 final CEMC slabs as ASE-written CIF files.
python run.py export --site bridge --temperature 298 --output exported_final
# Predict adsorption energies and activity without another MC calculation.
python run.py evaluate --site bridge --output evaluated_no_shift
# Apply the exact shifts used for Fig. 6 or S15.
python run.py evaluate --site bridge \
  --shifts cases/bridge/figure_mass_BE_shifts.json --output evaluated_shifted
```

`evaluate` writes `site_predictions.npz` and `activity.json`. Arrays contain the energy and element fractions of each adsorption site, for homogeneous, CEMC and within-layer-shuffled slabs. Each method's activity is the arithmetic mean of `exp(-abs(E-E_opt)/(k_B*298 K))`, not the mean of log(activity). Fe coordinates are `DeltaG_H = predicted_DeltaE_H - model.e_opt`; Pt coordinates are `DeltaE_OH`. A positive per-element shift is **subtracted** from the predicted BE (zone-1 average for multiatom sites).

Use `--activity-model /path/to/exported_model.txt` to substitute a compatible model with the same species, atom-index topology and supported site format. To evaluate another saved ensemble, add `--structures /path/to/structures.npz`. This allows several site models to use the same generated facet structures.

## Generate equiatomic slabs again

```bash
python run.py build
python run.py slabs --site bridge --ranks 4 --output generated_full
python run.py export --site bridge --structures generated_full/structures.npz \
  --temperature 298 --output generated_final_cifs
python run.py evaluate --site bridge --structures generated_full/structures.npz \
  --output evaluated_generated
# Dense snapshots for the temperature-dependent curves (10 K spacing):
python run.py slabs --site bridge --dense --ranks 4 --output generated_dense
```

The historical schedule holds 2023 K for 10,000 attempted swaps, then cools exponentially to 298 K in 100,000 attempted swaps. There are 1,000 atoms (200 of each element), ten layers and 100 atoms per layer. The default archive has 19 snapshots (2000, 1900, ..., 300, 298 K). Dense archives have 172 snapshots (10 K spacing plus 298 K). `--runs 1` is a quick reproducibility check; default `--runs 20` reproduces the entire ensemble. A fresh output directory is required to prevent mixed results.

The historical site cases used different seeds even on the same facet. Their configurations and saved ensembles are retained separately to reproduce the paper. For new calculations, generate one chosen case per facet and pass its `structures.npz` to the other compatible site model. This avoids repeated CEMC work but does not replace the historical ensembles used by the packaged figures. The shuffle seed also contains the historical site name for Fe cases.

## Files and ASE representation

- `src/`: original C++17 CE/MC backend and deterministic homogeneous-slab reconstruction code.
- `cases/SITE/inputs/`: CE coefficients/topology, activity model, geometry template, schedule and equiatomic composition.
- `cases/SITE/config.json`: portable configuration; `inputs/equi_atomic.ini` preserves the original absolute-path configuration for provenance only.
- `cases/SITE/structures.npz`: losslessly stored atomic numbers `Z[temperature,run,atom]`, `random_Z[run,atom]`, cell, fractional positions, PBC, energies and MC steps. The saved geometry here retains the original double precision.
- `cases/SITE/site_predictions_no_shift.npz`, `activity_no_shift.json`: cached predictions used by Fig. 4/S14 and independently checked against the source statistics.
- `dense_10K/`: original dense archive, its schedule and provenance. Its existing fractional-coordinate precision is preserved as supplied.

```python
import numpy as np
from ase import Atoms
with np.load('cases/bridge/structures.npz') as d:
    i = list(d['temperatures_K']).index(298)
    atoms = Atoms(numbers=d['Z'][i, 0], scaled_positions=d['scaled_positions'],
                  cell=d['cell_A'], pbc=d['pbc'])
```

The cell and positions are common to all snapshots; storing them once reduces archive size. Shuffled controls are reconstructed deterministically without cross-layer mixing. See `../../../analysis/equiatomic/README.md` and the root figure table for analysis commands and validation.
