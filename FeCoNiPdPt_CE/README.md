# FeCoNiPdPt Cluster Expansion Training and Evaluation Package

## 1. Contents and data

This package contains the inputs used by the original training configurations and saved models for three facets. Only structures listed in each `id_prop.csv` are included. CSV row order, target values, and CIF contents are preserved.
The target is the original CSV energy per metal atom, in **eV/metal atom**. These are cluster expansion (CE) models, separate from the H adsorption energy regression models.

| Facet | Bulk | Slab | Total | Fit | Holdout |
|---|---:|---:|---:|---:|---:|
| fcc(111) | 1100 | 700 | 1800 | 1620 | 180 |
| fcc(100) | 1100 | 1019 | 2119 | 1907 | 212 |
| fcc(110) | 1100 | 691 | 1791 | 1611 | 180 |

Bulk structures are included because they were inputs to the original models. The three datasets remain separate because target values can differ across facets even for the same identifier.

```text
FeCoNiPdPt_CE/
  README.md
  requirements.txt
  run_ce.py                    # Portable training, evaluation, and prediction
  code/fcc_ce/                 # Unmodified fcc-ce 0.1.2 source used originally
  dist/fcc_ce-0.1.2-py3-none-any.whl
  fcc111/                      # Same layout for fcc100 and fcc110
    train.yaml                 # Configuration with package-relative paths
    original_train.yaml        # Original configuration for provenance
    original_commands/         # Historical shell commands
    data/id_prop.csv           # Identifier and target in original row order
    data/structures/*.cif       # Structures used for training and evaluation
    data/reference_split.csv   # Original fit/holdout assignment
    reference/                 # Original model, coefficients, metrics, predictions
  results/                     # Results reproduced with this package
  provenance.json              # Sources, versions, input SHA-256 hashes
  validation.json              # Reproduction checks
```

## 2. Environment setup

The package was validated with Python 3.14. Example using Linux and Conda:

```bash
conda create -n feconipdpt-ce python=3.14 -y
conda activate feconipdpt-ce
cd /path/to/FeCoNiPdPt_CE
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Installing dependencies requires internet access or a repository containing the required packages. `requirements.txt` pins the validated versions. Other Python, library, or BLAS versions may produce small floating-point differences. Validation used one BLAS thread.

`run_ce.py` imports the bundled `code/fcc_ce` source directly; installing fcc-ce separately is optional. To use the original CLI as well:

```bash
python -m pip install --no-deps dist/fcc_ce-0.1.2-py3-none-any.whl
```

## 3. Evaluate the original saved models

```bash
python run_ce.py evaluate --facet all
# Evaluate one facet only:
python run_ce.py evaluate --facet 111
```

This recomputes features for all bundled structures, predicts with the original saved models, and compares predictions against the original `training_predictions.csv`. Results are written to each facet's directory, such as `results/fcc111/evaluation_reference/`. Inspect `metrics.json`, `saved_model_predictions.csv`, and `parity.png`.

Package execution does not read the original server paths. Absolute paths retained in original configurations and model metadata are provenance records.

## 4. Retrain with the original settings

```bash
python run_ce.py train --facet all
# Retrain one facet only:
python run_ce.py train --facet 100
```

Training uses the original 266 features, maximum pair shell 4, maximum triplet shell 2, random seed 42, a 10% holdout stratified by family, five-fold CV on the fit subset, and 31 ridge alpha candidates. CV selects alpha using the fit subset. A model trained only on that subset is evaluated on the holdout. The final saved model is then refitted on **all data** using the selected alpha.

Outputs are written to directories such as `results/fcc100/retrained/`:

- `model/metadata.json` and `model/coefficients.npz`: final refitted CE model
- `coefficients.csv`: feature coefficients
- `metrics.json`: CV, training_split, holdout, and full_refit metrics
- `training_predictions.csv`: final and validation predictions, with split labels
- `reproduction_check.json`: comparisons of row order, splits, alpha, coefficients, and predictions against the originals

Original `reference/` files are preserved. Repeated runs update the corresponding files under `results/`.

To evaluate the retrained models:

```bash
python run_ce.py evaluate --facet all --use-retrained
```

The original CLI can also be used. Its relative configuration paths are resolved from the current directory, so first enter the appropriate facet directory:

```bash
cd fcc111
fcc-ce train --config train.yaml --output ce_training_output
```

## 5. Predict new structures

```bash
python run_ce.py predict --facet 111 --inputs /path/to/structures --output predictions_111.csv
python run_ce.py predict --facet 100 --use-retrained --inputs new_structure.cif --output predictions_100.csv
```

`--facet` selects the facet model. The original model is used by default; `--use-retrained` selects the retrained model. Inputs may be a file or directory containing CIF, VASP, POSCAR, XYZ, or EXTXYZ structures. The output `prediction` is the original target in eV/metal atom; `total_energy` is its corresponding total target energy.

## 6. Interpret the evaluation metrics

In `training_predictions.csv`, `prediction` comes from the final model refitted on all data. `validation_prediction` comes from the model trained only on the fit subset. Original holdout metrics use `validation_prediction` on rows with `split=holdout`. Because the final model also includes those rows during refitting, its predictions on them are not independent test performance.

`evaluate` computes fresh saved-model predictions and recalculates holdout metrics from the stored `validation_prediction` column. To reproduce the holdout model training itself, run `train` first, then `evaluate --use-retrained`.

`parity.png` contains only holdout predictions from the model trained on the fit subset. Blue circles represent bulk structures and orange triangles represent slab structures. The annotation reports the holdout sample count, MAE and RMSE in meV/atom, and R-squared. Both axes show formation energy in eV/atom, with a dashed identity line. No full-data refit plot is generated.

Original holdout metrics, combining bulk and slab structures (eV/metal atom):

| Facet | MAE | RMSE | Selected alpha |
|---|---:|---:|---:|
| fcc(111) | 0.0042132451 | 0.0062462853 | 2.1544346900 |
| fcc(100) | 0.0098964600 | 0.0141802147 | 46.4158883361 |
| fcc(110) | 0.0100224167 | 0.0147536742 | 21.5443469003 |

Separate bulk/slab metrics are available under `by_family` in each `reference/metrics.json`.

Historical shell commands are preserved under each facet's `original_commands/`. They contain original working directories and placeholder paths. Use the `run_ce.py` commands above to run this portable package.

## 7. Validation performed during packaging

All three models were retrained using the bundled inputs and source. Input order, holdout splits, selected alpha, coefficients, final and validation predictions, and CV/evaluation metrics were compared against the originals. Evaluation also recomputed original saved-model predictions for every input structure. The prediction CLI was checked with bulk and slab structures for each facet.

Tolerances and observed differences are recorded in `validation.json` and each retraining output's `reproduction_check.json`. The archive includes the retrained models and evaluation results produced during validation.
