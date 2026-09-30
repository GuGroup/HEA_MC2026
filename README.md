# Paper reproduction: Rethinking the Role of Homogeneous Surface Models in High-Entropy Alloy Catalyst Screening

This repository contains code and data for reproducing **Table 1, Figures 1-6, and Figures S6-S15** of the manuscript. It includes H adsorption model training/testing, cluster expansion (CE) training/evaluation, uncertainty quantification Monte Carlo (UQMC), cluster expansion Monte Carlo (CEMC), and equiatomic slab analyses for FeCoNiPdPt and PtPdRhRuIr.

## Get the code and data

Clone the repository to download all four folders and this README together:

```bash
git clone https://github.com/GuGroup/HEA_MC2026.git
cd HEA_MC2026
```

Alternatively, select **Code > Download ZIP** on the repository page and extract that single repository download. Open the extracted repository folder before following the instructions below. All required package files are stored directly in this repository; separate Release downloads are not required.

The four folders contain English READMEs with detailed instructions. No connection to the original calculation server is required to use the supplied data and code.

## 1. Choose a folder

| Folder | Contents | Main entry point |
|---|---|---|
| [FeCoNiPdPt_H_adsorption](FeCoNiPdPt_H_adsorption/) | H-adsorbed structures, DFT adsorption energies, regression training code, fitted models, test data and parity plots for five facet/site combinations | `evaluate_all.py` and each site's `train_model.py` |
| [FeCoNiPdPt_CE](FeCoNiPdPt_CE/) | CE training data, original source, saved models, retraining and holdout evaluation for FeCoNiPdPt fcc(111), fcc(100), and fcc(110) | `run_ce.py` |
| [PtPdRhRuIr_CE](PtPdRhRuIr_CE/) | CE training data, original source, saved model, retraining and holdout evaluation for PtPdRhRuIr fcc(111) | `run_ce.py` |
| [HEA_UQMC_CEMC_figures](HEA_UQMC_CEMC_figures/) | Table 1 trial scores and reproduction code, numeric figure data, plotting code, simulation sources and model exports, equiatomic structures, and four facet workflows | `Table_1/reproduce.py`, `reproduce_all_figures.py`, and workflow-specific `run.py` |

The adsorption regression models predict adsorption energy in eV. The CE models predict the original structure-energy target in eV per metal atom. These are different models with different purposes. The figure folder already contains the exported CE and adsorption inputs needed by its simulation backends; retraining the other three packages is not required to reproduce the figures and table or run the supplied workflows.

## 2. Repository layout and environments

```text
HEA_MC2026/
  README.md
  FeCoNiPdPt_H_adsorption/
  FeCoNiPdPt_CE/
  PtPdRhRuIr_CE/
  HEA_UQMC_CEMC_figures/
    Table_1/
    Fig_1/ ... Fig_6/
    Fig_S6/ ... Fig_S15/
    workflows/
```

The examples below use Linux/Conda. On Windows, use a suitable Linux environment such as WSL for the C++/MPI simulation commands. Replace `/path/to/HEA_MC2026` with the actual location of your cloned or downloaded repository (the downloaded folder may be named `HEA_MC2026-main`). Each command block identifies the package or workflow directory from which it should be run.

Use the environment setup in each section below. Separate environments avoid conflicts between pinned dependency versions. Adsorption regression was validated with Python 3.12.7; CE and figure packages were validated with Python 3.14. Installing dependencies requires internet access or a local package mirror. Small numerical or rendering differences may occur with other library/compiler versions.

Python alone is sufficient for plotting, Table 1 reproduction and model evaluation after installing the requirements. CEMC generation additionally requires a C++17 compiler and MPI (`g++`, `mpicxx`, and `mpirun` on `PATH`). Compiled simulation executables are built locally from the included sources.

## 3. FeCoNiPdPt H adsorption models

See [FeCoNiPdPt_H_adsorption/README.md](FeCoNiPdPt_H_adsorption/README.md) for the detailed guide.

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
cd /path/to/HEA_MC2026/FeCoNiPdPt_H_adsorption
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

See [FeCoNiPdPt_CE/README.md](FeCoNiPdPt_CE/README.md) for the detailed guide.

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
cd /path/to/HEA_MC2026/FeCoNiPdPt_CE
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

See [PtPdRhRuIr_CE/README.md](PtPdRhRuIr_CE/README.md) for the detailed guide.

This package has the same portable CE interface, for PtPdRhRuIr fcc(111). It contains 1,757 structures: 1,100 bulk and 657 slab structures, with 1,581 fit and 176 holdout structures. Original source, model, training settings, splits, predictions and validation records are included.

```bash
conda create -n ptpdrhruir-ce python=3.14 -y
conda activate ptpdrhruir-ce
cd /path/to/HEA_MC2026/PtPdRhRuIr_CE
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python run_ce.py evaluate --facet 111
python run_ce.py train --facet 111
python run_ce.py evaluate --facet 111 --use-retrained
python run_ce.py predict --facet 111 --inputs /path/to/structures --output predictions.csv
```

Outputs are under `results/fcc111/`. The holdout/refit distinction and parity-plot conventions described for FeCoNiPdPt also apply here. The original holdout MAE and RMSE are approximately 1.763 and 2.353 meV/atom.

## 6. UQMC, CEMC, manuscript figures and Table 1

See [HEA_UQMC_CEMC_figures/README.md](HEA_UQMC_CEMC_figures/README.md) for the detailed guide.

### Reproduce the figures from the supplied data

```bash
conda create -n hea-figures python=3.14 -y
conda activate hea-figures
cd /path/to/HEA_MC2026/HEA_UQMC_CEMC_figures
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

### Reproduce Table 1 from the supplied trial scores

Run the following from the figure package directory. Table reproduction only requires NumPy; it does not require new CEMC or UQMC calculations.

```bash
cd /path/to/HEA_MC2026/HEA_UQMC_CEMC_figures
python -m pip install -r Table_1/requirements.txt
python Table_1/reproduce.py
```

[Table_1](HEA_UQMC_CEMC_figures/Table_1/) contains paired CEMC/homogeneous scores for 10,000 trials in each of 12 datasets, the calculation script, the manuscript reference, and generated tables. Outputs are saved under `Table_1/outputs/` as CSV, Markdown and HTML, together with `validation.json`. The figure-only launcher does not run the table script.

All 52 displayed improvement probabilities match the supplied manuscript, including the Average row. Two percentile endpoints differ: the Study 1 delta tau upper endpoint is 0.38 (manuscript 0.39), and the Study 2 ML4 delta CRPS lower endpoint is -0.03 (manuscript -0.04). The calculated values are preserved and the comparison is documented in [Table_1/README.md](HEA_UQMC_CEMC_figures/Table_1/README.md).

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
cd /path/to/HEA_MC2026/HEA_UQMC_CEMC_figures/workflows/PtPdRhRuIr_fcc111
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
cd /path/to/HEA_MC2026/HEA_UQMC_CEMC_figures/workflows/FeCoNiPdPt_fcc100
python run.py build
python run.py slabs --case hollow --ranks 48
python run.py activity --case hollow --ranks 48
python run.py activity --case bridge --ranks 48
python run.py random --case hollow --ranks 48
python run.py random --case bridge --ranks 48
```

The activity commands reuse `outputs/slabs_shared_facet/`. Different composition datasets, facets, schedules, CE models or composition uncertainty settings require their own trajectories. Study 1 has a separate implementation and packed format under `workflows/PtPdRhRuIr_fcc111/Study1/`; follow its README rather than the ML1-ML6 commands.

Full production slab/activity collections are not included because they occupy hundreds of GB per trajectory set. This folder includes the figure and Table 1 data, simulation inputs and real small slab samples. Full UQMC regeneration requires substantial storage and computing time. Read the package README for shard boundaries, restart behavior, model overrides and postprocessing.

### Run or reuse equiatomic CEMC calculations

Each of the four facet folders also contains an independent `equi-atomic/` workflow. These fixed-composition calculations use **20 runs**, not the 10,000 uncertainty trials above. The original ensembles and compact dense-temperature snapshots are included.

Example for FeCoNiPdPt fcc(100), starting from its `equi-atomic/` directory:

```bash
cd /path/to/HEA_MC2026/HEA_UQMC_CEMC_figures/workflows/FeCoNiPdPt_fcc100/equi-atomic

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
