# PtPdRhRuIr_fcc111 UQMC and CEMC workflow

This folder contains the original facet backend and the exact exported CE, schedule, adsorption model, composition, and experimental inputs needed to run it. Compile on Linux with a C++17 MPI compiler. Run commands from this folder or invoke `run.py` by its full path.

```bash
python run.py build
python run.py slabs --case ML1 --ranks 48
python run.py activity --case ML1 --ranks 48
python run.py random --case ML1 --ranks 48
```

Available cases: ML1, ML2, ML3, ML4, ML5, ML6. Each ML has its own composition list and slab directory. Study 1 has its own subfolder and README.

Use `--slab-dir /path/to/existing/slabs` with `activity` to reuse existing trajectories. Use `--activity-model`, `--be-mean`, and `--be-sigma` to evaluate a compatible alternative model without another CEMC run. `--output` selects a separate empty evaluation directory. Configurations in `configs/` contain package-relative input paths. `provenance/` retains the original configuration for reference.

A small, real sample is included. To test the stored slab without performing a full calculation:

```bash
python run.py activity --case ML1 --sample --output outputs/sample_check
python run.py slabs --case ML1 --sample --output outputs/sample_generation
```

The sample contains trial 0, composition 0, run 0, and all 18 temperatures. The annealing schedule and model are unchanged. Compare with `sample/expected_activity.json` and `sample/slabs/`. Full production slab and activity collections are not included due to their size. See the root README for the complete figure and postprocessing workflow.

The original double-precision homogeneous log-mean generator is also retained:

```bash
mpicxx -O3 -std=c++17 -DUSE_MPI -fopenmp src/recover_random_logmean.cpp -o build/recover_random_logmean
# First run a wrapper command to create the absolute-path runtime configuration.
./build/recover_random_logmean outputs/runtime_configs/ML1_random.ini outputs/ML1_random_logmean 8
```

Use a production runtime configuration, not one ending in `_sample.ini`, for the historical 10,000-trial calculation. This utility writes mean(log(activity)) CSV shards directly in double precision. The general `random` wrapper instead writes float32 individual activities, which can cause small rounding differences in downstream metrics.

## Equiatomic calculations

See [equi-atomic/README.md](equi-atomic/README.md) for the independent 20-run equiatomic CEMC calculations and Figures 3-6, S13-S15.
