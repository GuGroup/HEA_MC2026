# PtPdRhRuIr fcc(111) Cluster Expansion Training and Evaluation Package

This package contains the CIF structures and `id_prop.csv` energy targets used by the original configuration and saved model, together with the code, model, and evaluation results. Only the 1,757 structures listed in the CSV are included. Original file contents and CSV row order are preserved.

## Data and evaluation procedure

- 1,100 bulk structures and 657 slab111 structures: 1,757 total
- Fit subset: 1,581 structures (990 bulk and 591 slab)
- Holdout subset: 176 structures (110 bulk and 66 slab)
- Target: original `id_prop.csv` energy per metal atom, in eV/metal atom
- 266 features, maximum pair shell 4, maximum triplet shell 2
- Random seed 42, 10% holdout stratified by family, five-fold CV on the fit subset, and 31 ridge alpha candidates
- Original selected alpha: 1e-8

The original procedure selects alpha by CV, evaluates the holdout using a model trained only on the fit subset, and then refits the final saved model on all data.

## Directory layout

```text
PtPdRhRuIr_CE/
  README.md
  requirements.txt
  run_ce.py
  code/fcc_ce/                  # Original fcc-ce source used for training
  dist/fcc_ce-0.1.2-py3-none-any.whl
  fcc111/
    train.yaml                 # Configuration with package-relative paths
    original_train.yaml        # Original configuration for provenance
    original_commands/         # Historical shell commands
    data/id_prop.csv
    data/structures/*.cif
    data/reference_split.csv
    reference/                 # Original model, coefficients, metrics, predictions
  results/fcc111/
    retrained/                 # Retrained model and reproduction comparisons
    evaluation_reference/      # Original saved-model evaluation and parity plot
  provenance.json              # Sources, versions, structure SHA-256 hashes
  validation.json              # Reproduction checks
```

## Environment setup

The package was validated with Python 3.14 and the versions pinned in `requirements.txt`.

```bash
conda create -n ptpdrhruir-ce python=3.14 -y
conda activate ptpdrhruir-ce
cd /path/to/PtPdRhRuIr_CE
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Installing dependencies requires internet access or a repository containing the required packages. `run_ce.py` directly imports the bundled `code/fcc_ce` source, so a separate fcc-ce installation is optional. To use the original CLI as well:

```bash
python -m pip install --no-deps dist/fcc_ce-0.1.2-py3-none-any.whl
```

## Evaluate the original model

```bash
python run_ce.py evaluate --facet 111
```

This recomputes features for every bundled structure and compares the original saved-model predictions with the original prediction CSV. Results are written to `results/fcc111/evaluation_reference/`. Inspect `metrics.json`, `saved_model_predictions.csv`, and `parity.png`.

## Retrain and evaluate the retrained model

```bash
python run_ce.py train --facet 111
python run_ce.py evaluate --facet 111 --use-retrained
```

Training uses the original settings and compares input order, holdout splits, alpha, coefficients, and final/validation predictions. It writes `model/metadata.json`, `model/coefficients.npz`, `coefficients.csv`, `metrics.json`, `training_predictions.csv`, and `reproduction_check.json` under `results/fcc111/retrained/`.

Original `reference/` files are preserved. Repeated runs update the corresponding files under `results/`.

To use the original CLI directly, enter the facet directory because relative configuration paths are resolved from the current directory:

```bash
cd fcc111
fcc-ce train --config train.yaml --output ce_training_output
```

## Predict new structures

```bash
python run_ce.py predict --facet 111 --inputs /path/to/new_structures --output predictions.csv
python run_ce.py predict --facet 111 --use-retrained --inputs new_structure.cif --output predictions_retrained.csv
```

The original model is used by default; `--use-retrained` selects the retrained model. Inputs may be a file or directory containing CIF, VASP, POSCAR, XYZ, or EXTXYZ structures. The output `prediction` is the original target in eV/metal atom; `total_energy` is its corresponding total target energy.

The wrapper resolves bundled resources relative to its own location and can be invoked from another working directory. Original server paths retained in configurations and metadata are provenance records and are not accessed during package execution.

## Interpret the metrics

Original holdout performance combining bulk and slab structures: MAE 0.0017630446 and RMSE 0.0023533581 eV/metal atom. For the slab111 holdout alone (N=66): MAE 0.0024264165 and RMSE 0.0030597039 eV/metal atom.

In `training_predictions.csv`, `prediction` comes from the final model trained on all data. `validation_prediction` comes from the model trained only on the fit subset. Independent holdout performance must be calculated from `validation_prediction` on rows with `split=holdout`. The final model includes holdout rows during refitting, so its predictions on those rows are not independent test performance.

`evaluate` computes fresh final-model predictions and recalculates holdout metrics from the stored `validation_prediction` column. To repeat the holdout model training itself, run `train` followed by `evaluate --use-retrained`.

`parity.png` contains only holdout predictions from the model trained on the fit subset. Blue circles represent bulk structures and orange triangles represent slab structures. The annotation reports the holdout sample count, MAE and RMSE in meV/atom, and R-squared. Both axes show formation energy in eV/atom, with a dashed identity line. No full-data refit plot is generated.

Other Python, library, or BLAS environments may produce small floating-point differences.

Historical shell commands are preserved under `fcc111/original_commands/`. They contain original working directories and placeholder paths. Use the `run_ce.py` commands above to run this portable package.

## Validation performed during packaging

The bundled source was checked against the installed wheel, and all input file hashes were verified. The fcc(111) model was fully retrained using the bundled inputs and source. Input order, holdout splits, selected alpha, coefficients, final and validation predictions, and CV/evaluation metrics were compared against the originals. Evaluation also recomputed original saved-model predictions for every input structure. The prediction CLI was checked with bulk and slab structures.

Tolerances and observed differences are recorded in `validation.json` and the retraining output's `reproduction_check.json`. The archive includes the retrained model and evaluation results produced during validation.
