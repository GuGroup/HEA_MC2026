# HEA Adsorption Models, Cluster Expansion, and Figure Reproduction

This collection contains four archives for FeCoNiPdPt and PtPdRhRuIr high-entropy alloy calculations. It provides the data and code for H adsorption model training/testing, cluster expansion (CE) training/evaluation, uncertainty-aware Monte Carlo (UQMC), cluster expansion Monte Carlo (CEMC), and reproduction of the manuscript figures.

Download the four ZIP files from the GitHub release associated with this repository. Extract each ZIP once; it creates its own top-level directory. The packages contain English READMEs with more detailed instructions. No connection to the original calculation server is required.

## 1. Choose a package

| Archive | Approximate ZIP size | Contents | Main entry point |
|---|---:|---|---|
| `FeCoNiPdPt_H_adsorption.zip` | 9.3 MB | H-adsorbed structures, DFT adsorption energies, regression training code, fitted models, test data and parity plots for five facet/site combinations | `evaluate_all.py` and each site's `train_model.py` |
| `FeCoNiPdPt_CE.zip` | 6.1 MB | CE training data, original source, saved models, retraining and holdout evaluation for FeCoNiPdPt fcc(111), fcc(100), and fcc(110) | `run_ce.py` |
| `PtPdRhRuIr_CE.zip` | 1.9 MB | CE training data, original source, saved model, retraining and holdout evaluation for PtPdRhRuIr fcc(111) | `run_ce.py` |
| `HEA_UQMC_CEMC_figures.zip` | 207.4 MB | Numeric figure data, plotting code, simulation sources and model exports, equiatomic structures, and four facet workflows | `reproduce_all_figures.py` and workflow-specific `run.py` |

The adsorption regression models predict adsorption energy in eV. The CE models predict the original structure-energy target in eV per metal atom. These are different models with different purposes. The figure package already contains the exported CE and adsorption inputs needed by its simulation backends; retraining the first three packages is not required to reproduce its figures or run its supplied workflows.

Earlier archives named `fcc_H_training_bundle.zip` and `FeCoNiPdPt_fcc111_H.zip` are narrower versions superseded by `FeCoNiPdPt_H_adsorption.zip` and are not required for this collection.

## 2. Extract the archives and prepare environments

The examples below use Linux/Conda. Run extraction in the directory where the ZIP files were downloaded:

```bash
unzip FeCoNiPdPt_H_adsorption.zip
unzip FeCoNiPdPt_CE.zip
unzip PtPdRhRuIr_CE.zip
unzip HEA_UQMC_CEMC_figures.zip
```

The resulting layout is:

```text
README.md
FeCoNiPdPt_H_adsorption/
FeCoNiPdPt_CE/
PtPdRhRuIr_CE/
HEA_UQMC_CEMC_figures/
```

Use the environment setup in each section below. Separate environments avoid conflicts between pinned dependency versions. Adsorption regression was validated with Python 3.12.7; CE and figure packages were validated with Python 3.14. Installing dependencies requires internet access or a local package mirror. Small numerical or rendering differences may occur with other library/compiler versions.

Python alone is sufficient for plotting and model evaluation after installing the requirements. CEMC generation additionally requires a C++17 compiler and MPI (`g++`, `mpicxx`, and `mpirun` on `PATH`). Compiled simulation executables are built locally from the included sources.

## 3. FeCoNiPdPt H adsorption models

Detailed guide: [FeCoNiPdPt_H_adsorption/README.md](FeCoNiPdPt_H_adsorption/README.md).

### Contents

```text
FeCoNiPdPt_H_adsorption/
  requirements.txt
  evaluate_all.py
  fcc111/fcc_hollow/{training,testing}/
  fcc111/top/{training,testing}/
  fcc100/hollow/{training,testing}/
  fcc100/bridge/{training,testing}/
  fcc110/bridge/{training,testing}/
```

`fcc110/bridge` denotes the original long-bridge site. Each `training/` directory contains `training_data.json`, `train_model.py`, and `model.json`; each `testing/` directory contains `test_data.json`, `evaluate_model.py`, and evaluation outputs.

Each data record contains its original index, H-adsorbed structure encoded as ASE Atoms, and adsorption energy in eV. Only the indices in the corresponding original energy CSV are included, in CSV order. Indices are local to each facet/site/split. Original feature definitions, atom order, geometry rules, and no-intercept regression are retained. No BE shifts are applied in this package.

### Install and evaluate the supplied models

```bash
conda create -n hea-h-adsorption python=3.12.7 -y
conda activate hea-h-adsorption
cd /path/to/FeCoNiPdPt_H_adsorption
python -m pip install -r requirements.txt
python evaluate_all.py
```

Evaluation uses the existing model files and does not train on test data. For one site:

```bash
python fcc100/hollow/testing/evaluate_model.py
```

### Retrain all five models and reevaluate

```bash
python fcc111/fcc_hollow/training/train_model.py
python fcc111/top/training/train_model.py
python fcc100/hollow/training/train_model.py
python fcc100/bridge/training/train_model.py
python fcc110/bridge/training/train_model.py
python evaluate_all.py
```

Inspect the models in `training/` and the predictions, metrics and plots in `testing/`. Training regenerates the corresponding model/output files; keep a separate copy if you want to preserve a previous fit.

The original fcc(100) hollow parity plot selected test points with absolute error at most 0.1 eV. The original fcc(110) bridge plot excluded the three largest errors. Both historical selections are reproduced explicitly. All test records remain available, and `metrics.json` distinguishes `test_all` from `test_original_plot`. Use the full-test metrics and `parity_all_test.png` for complete-test performance; `parity_original.png` preserves the historical plot.

Read a stored ASE structure with:

```python
from pathlib import Path
from ase.io.jsonio import decode

record = decode(Path('fcc100/hollow/training/training_data.json').read_text())['records'][0]
index = record['index']
atoms = record['atoms']
energy_eV = record['adsorption_energy_eV']
```

## 4. FeCoNiPdPt cluster expansion models

Detailed guide: [FeCoNiPdPt_CE/README.md](FeCoNiPdPt_CE/README.md).

### Contents

The package contains original `fcc-ce` source, CIF structures, target tables, train/holdout assignments, training configurations, saved CE models, and reproduced evaluation outputs for three facets. Only structures listed in each original `id_prop.csv` are included.

| Facet | Bulk structures | Slab structures | Total | Holdout |
|---|---:|---:|---:|---:|
| fcc(111) | 1,100 | 700 | 1,800 | 180 |
| fcc(100) | 1,100 | 1,019 | 2,119 | 212 |
| fcc(110) | 1,100 | 691 | 1,791 | 180 |

Facet datasets are separate because target values can differ even for the same identifier. The package retains the original 266 features, random seed 42, family-stratified 10% holdout, five-fold CV and ridge-regularization selection.

### Install, evaluate and retrain

```bash
conda create -n feconipdpt-ce python=3.14 -y
conda activate feconipdpt-ce
cd /path/to/FeCoNiPdPt_CE
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Evaluate the original saved models.
python run_ce.py evaluate --facet all

# Repeat training with the original settings, then evaluate the new models.
python run_ce.py train --facet all
python run_ce.py evaluate --facet all --use-retrained
```

Replace `all` with `111`, `100`, or `110` to process one facet. `run_ce.py` imports the included source directly; installing the bundled wheel separately is optional.

Results are written below `results/fcc111/`, `results/fcc100/`, and `results/fcc110/`, including model files, `metrics.json`, prediction tables and `parity.png`. Retraining writes `retrained/reproduction_check.json`. Original `reference/` files are preserved.

Predict new structures with:

```bash
python run_ce.py predict --facet 111 --inputs /path/to/structures --output predictions_111.csv
```

Add `--use-retrained` to select the retrained model. Input can be a structure file or a directory of supported structure files.

### Interpret the parity plots

Holdout plots use predictions from the model trained only on the fit subset. They show bulk structures as blue circles and slabs as orange triangles, with sample count, MAE, RMSE, and R-squared. MAE/RMSE annotations are in meV/atom; axes are in eV/atom. No full-data-refit parity plot is generated.

The final saved CE model is refitted on all data after hyperparameter selection. Its predictions on former holdout rows are therefore not independent test results. In prediction tables, `validation_prediction` is used for holdout performance, while `prediction` belongs to the final refitted model. `evaluate` recalculates holdout metrics from stored validation predictions; run `train` to repeat the holdout-model fit itself.

## 5. PtPdRhRuIr fcc(111) cluster expansion model

Detailed guide: [PtPdRhRuIr_CE/README.md](PtPdRhRuIr_CE/README.md).

This package has the same portable CE interface, for PtPdRhRuIr fcc(111). It contains 1,757 structures: 1,100 bulk and 657 slab structures, with 1,581 fit and 176 holdout structures. Original source, model, training settings, splits, predictions and validation records are included.

```bash
conda create -n ptpdrhruir-ce python=3.14 -y
conda activate ptpdrhruir-ce
cd /path/to/PtPdRhRuIr_CE
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python run_ce.py evaluate --facet 111
python run_ce.py train --facet 111
python run_ce.py evaluate --facet 111 --use-retrained
python run_ce.py predict --facet 111 --inputs /path/to/structures --output predictions.csv
```

Outputs are under `results/fcc111/`. The holdout/refit distinction and parity-plot conventions described for FeCoNiPdPt also apply here. The original holdout MAE and RMSE are approximately 1.763 and 2.353 meV/atom.

## 6. UQMC, CEMC and manuscript figures

Detailed guide: [HEA_UQMC_CEMC_figures/README.md](HEA_UQMC_CEMC_figures/README.md).

### Reproduce the figures from the supplied data

```bash
conda create -n hea-figures python=3.14 -y
conda activate hea-figures
cd /path/to/HEA_UQMC_CEMC_figures
python -m pip install -r requirements.txt
python reproduce_all_figures.py

# Or reproduce one figure only:
python Fig_3/plot.py
```

This plotting step does not run CEMC or the 10,000-trial UQMC calculation. Each figure folder contains its data, plotting code, English README and outputs. Figures can be rendered without generating new structures.

| Figure folders | Contents |
|---|---|
| `Fig_1`, `Fig_2` | Study 1, Study 2 ML4 and Study 3 fcc(100) hollow: metric distributions and selected activity maps |
| `Fig_3` | Equiatomic surface composition and Warren-Cowley order versus temperature |
| `Fig_4` | BE densities for homogeneous, CEMC and layer-shuffled Pt fcc(111)/Fe fcc(100) hollow slabs |
| `Fig_5` | Homogeneous BE distributions and activity volcanoes |
| `Fig_6` | Element-conditioned BE probability masses and containment |
| `Fig_S6`-`Fig_S9` | Metric distributions and activity maps for additional ML/site cases |
| `Fig_S10` | Best-performing-trial activity maps |
| `Fig_S11`, `Fig_S12` | Continuous recall analyses |
| `Fig_S13` | Final layer-pair compositions for all four facets |
| `Fig_S14`, `Fig_S15` | Additional Fe facet/site BE densities and conditional probability masses |

The equiatomic figures have PNG and vector PDF outputs. See each figure's README for output names and statistical conventions. Composite layouts are regenerated from numeric data; typography and spacing can differ from the embedded Word figures.

### Run the production UQMC/CEMC workflow

Four physical facet workflows are included:

| Workflow folder under `workflows/` | Valid production `--case` values |
|---|---|
| `PtPdRhRuIr_fcc111` | `ML1`, `ML2`, `ML3`, `ML4`, `ML5`, `ML6` |
| `FeCoNiPdPt_fcc111` | `fcc` (fcc hollow), `top` |
| `FeCoNiPdPt_fcc110` | `bridge` (long bridge) |
| `FeCoNiPdPt_fcc100` | `hollow`, `bridge` |

For Study 2 ML4:

```bash
cd /path/to/HEA_UQMC_CEMC_figures/workflows/PtPdRhRuIr_fcc111
python run.py build
python run.py slabs --case ML4 --ranks 48
python run.py activity --case ML4 --ranks 48
python run.py random --case ML4 --ranks 48
```

- `slabs` generates CEMC structures for **10,000 trials (IDs 0-9999)** under the supplied production settings. Each trial uses all compositions in the selected dataset and 20 independent runs per composition.
- `--ranks 48` distributes the calculation across 48 MPI processes. The total remains 10,000 trials.
- `activity` reuses saved CEMC slabs and applies the adsorption model and BE uncertainty.
- `random` evaluates the homogeneous ensembles reconstructed from deterministic seeds.

A smaller 40-trial generation run can be requested explicitly:

```bash
python run.py slabs --case ML4 --ranks 4 --trial-start 0 --trial-end 40 \
  --output outputs/ML4_first40_slabs
```

The end index is exclusive. This still uses all ML4 compositions and 20 runs each. For the smaller bundled one-composition/one-run verification example, use the Pt **ML1** sample:

```bash
python run.py activity --case ML1 --sample --output outputs/sample_ML1_activity
```

For FeCoNiPdPt fcc(100), generate a single facet trajectory set and evaluate both site models:

```bash
cd /path/to/HEA_UQMC_CEMC_figures/workflows/FeCoNiPdPt_fcc100
python run.py build
python run.py slabs --case hollow --ranks 48
python run.py activity --case hollow --ranks 48
python run.py activity --case bridge --ranks 48
python run.py random --case hollow --ranks 48
python run.py random --case bridge --ranks 48
```

The activity commands reuse `outputs/slabs_shared_facet/`. Different composition datasets, facets, schedules, CE models or composition uncertainty settings require their own trajectories. Study 1 has a separate implementation and packed format under `workflows/PtPdRhRuIr_fcc111/Study1/`; follow its README rather than the ML1-ML6 commands.

Full production slab/activity collections are not included because they occupy hundreds of GB per trajectory set. The archive includes all figure data, simulation inputs and real small slab samples. Full UQMC regeneration requires substantial storage and computing time. Read the package README for shard boundaries, restart behavior, model overrides and postprocessing.

### Run or reuse equiatomic CEMC calculations

Each of the four facet folders also contains an independent `equi-atomic/` workflow. These fixed-composition calculations use **20 runs**, not the 10,000 uncertainty trials above. The original ensembles and compact dense-temperature snapshots are included.

Example for FeCoNiPdPt fcc(100), starting from its `equi-atomic/` directory:

```bash
cd /path/to/HEA_UQMC_CEMC_figures/workflows/FeCoNiPdPt_fcc100/equi-atomic

# Reuse the supplied structures; no MC rerun is needed.
python run.py export --site hollow --temperature 298 --output exported_final
python run.py evaluate --site hollow --output evaluated_no_shift

# Regenerate 20 runs using the original equiatomic configuration.
python run.py build
python run.py slabs --site hollow --ranks 4 --output generated_full
python run.py evaluate --site hollow --structures generated_full/structures.npz \
  --output evaluated_generated

# Optional dense temperature snapshots at 10 K spacing.
python run.py slabs --site hollow --dense --ranks 4 --output generated_dense
```

Equiatomic `--site` values are `top` for Pt fcc(111); `hollow`/`top` for Fe fcc(111); `bridge` for Fe fcc(110); and `hollow`/`bridge` for Fe fcc(100). In particular, the production Fe fcc(111) case is named `fcc`, while its equiatomic site is named `hollow`.

Original equiatomic site cases used different seeds. Those historical ensembles are preserved for exact numeric figure reproduction. For new calculations, a common facet ensemble can be reused with another compatible site model using `--structures`; see the facet README for `--activity-model` and `--shifts` options. Simulation backends require their exported text model format; an adsorption training `model.json` cannot be passed directly as that export.

From the figure package root, recompute the equiatomic numerical inputs and comparisons with:

```bash
python analysis/equiatomic/recompute.py
```

This evaluates stored structures and reconstructs the deterministic layer-shuffled controls; it does not rerun CEMC. Fig. 4/5/S14 use zero BE shifts, whereas Fig. 6/S15 preserve the nonzero shifts used by their original source plots. Fig. 3 uses the topmost layer; Fig. S13 pools symmetric layer pairs. Exact definitions and shifts are documented inside the package.

## 7. Reproducibility and outputs

The adsorption and CE packages contain `validation.json`; the figure package contains `validation/` records and `manifest_sha256.json`. Original reference data and provenance are retained. The packaging checks include model retraining/evaluation, prediction comparisons, historical test selection checks, numeric figure comparisons, CEMC sample regeneration, and execution after relocation.

All six historical equiatomic facet/site cases were checked using one full 19-temperature MC trajectory each; their atomic identities matched the originals exactly. Equiatomic figure quantities matched the source tables within floating-point precision (largest recorded difference below 5e-14). These checks do not imply a rerun of every 10,000-trial production calculation.

Follow the README inside each extracted package for complete input/output definitions, reference metrics and known source inconsistencies. In particular, the Fe fcc(100) layer-shuffled activity in Fig. 4 is approximately 0.1621 (displayed 0.16); the manuscript prose value 0.038 differs from the supplied figure and source calculation.
