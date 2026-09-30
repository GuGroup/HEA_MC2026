# Study 1 PtPdRhRuIr fcc(111) workflow

Source: `/home/ktg0829/project/HEA/CEMC/UQMC_new_composition_shift`.
This is the 1,400-composition Study 1 dataset. Its original implementation and packed format differ from the ML1-ML6 implementation one directory above. It remains part of the same PtPdRhRuIr fcc(111) physical facet.

## Build and generate the slab trajectories

```bash
python run.py build
# Five independent trial shards: index 0, 1, 2, 3, or 4.
python run.py slabs --shard-index 0 --ranks 48
python run.py slabs --shard-index 1 --ranks 48
python run.py slabs --shard-index 2 --ranks 48
python run.py slabs --shard-index 3 --ranks 48
python run.py slabs --shard-index 4 --ranks 48
```

Each shard uses its original `trial_start=index`, `trial_stride=5` assignment within 10,000 total trials. Output is `outputs/slabs/shard_XX/packed_atoms/`. Each trial file contains the homogeneous initial slab followed by the 18 CEMC temperature states, in composition/run/state/atom order. The original `metadata.json` defines the 3-bit species encoding. This format must not be passed to the header-based `C3SLAB01` reader used by the other workflows.

The source has one portability-only C++ change: the `simple_fs::path` type is fully qualified in `directory_entry` to compile with GCC. No numerical or random-number logic is changed. Model coefficients, CE topology, original composition order, and annealing schedule are supplied in `inputs/`. `simulation_experimental_activity.json` is the exact parent-directory experimental input referenced by the original simulation configuration; plotting uses the separately supplied log-scaled experimental input.

## Reevaluate the same slabs with an adsorption model

```bash
python run.py activity --workers 4
# Reuse an existing packed slab collection with an alternative compatible model:
python run.py activity --slab-root /path/to/results \
  --activity-model /path/to/activity_model_oh.txt \
  --be-mean 0.04233 --be-sigma 0.2604 --output outputs/new_oh_model
```

The activity command creates links to the existing packed slabs under the new output directory and writes only new activity results there. It does not rerun CEMC or change the linked packed data. Use a fresh output directory for a new model; completed activity trials are skipped by the original calculator. `--workers` controls parallel trial processing.

The BE shift exporter reproduces deterministic trial-specific shifts. `calculate_packed_activities.py` computes log slab activity with a numerically stable log-sum-exp operation and then the run mean and sample standard deviation. Outputs use the original uncompressed Zarr layout, read by the bundled scripts without requiring the zarr Python package.

## Score regenerated activities

```bash
python run.py score --slab-root outputs/activity --output outputs/scores
```

The wrapper scores all 1,400 compositions, matching the historical histogram tables. The underlying script also retains its center-cross exclusion mode; omit `--include-center-cross` when calling that script directly to score the 1,317-composition subset. These are distinct analysis scopes. The supplied figure inputs preserve the actual original scope of each panel.

Study 1 uses Gaussian CRPS with the run standard deviation. The other datasets use a different empirical-distribution CRPS implementation. Do not exchange their scoring routines. The original scripts for continuous-KDE representative selection, global best-score selection, clipping, and map rendering are retained under `scripts/`. For direct figure reproduction with no simulation, use the root `Fig_1`, `Fig_2`, `Fig_S10`, and `Fig_S12` folders.

## Small verification sample

```bash
python run.py activity --sample --output outputs/sample_activity
python run.py slabs --sample --output outputs/sample_slabs
```

This sample is original trial 280, composition 0, all 20 runs, with the homogeneous state and all 18 CEMC temperatures. It is small enough to run locally while preserving the original annealing schedule. The corresponding packed file and reference log-activity means/standard deviations are under `sample/`.

Full production trajectories are not included. Selected activity arrays needed to reproduce the paper, the score tables, actual sample trajectories, and original plotting inputs are included.
