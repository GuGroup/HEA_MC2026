# FeCoNiPdPt_fcc110 UQMC and CEMC workflow

This folder contains the original facet backend and the exact exported CE, schedule, adsorption model, composition, and experimental inputs needed to run it. Compile on Linux with a C++17 MPI compiler. Run commands from this folder or invoke `run.py` by its full path.

```bash
python run.py build
python run.py slabs --case bridge --ranks 48
python run.py activity --case bridge --ranks 48
python run.py random --case bridge --ranks 48
```

Available cases: bridge. All cases share outputs/slabs_shared_facet/. Generate this once, then evaluate each site model using the same slabs.

Use `--slab-dir /path/to/existing/slabs` with `activity` to reuse existing trajectories. Use `--activity-model`, `--be-mean`, and `--be-sigma` to evaluate a compatible alternative model without another CEMC run. `--output` selects a separate empty evaluation directory. Configurations in `configs/` contain package-relative input paths. `provenance/` retains the original configuration for reference.

A small, real sample is included. To test the stored slab without performing a full calculation:

```bash
python run.py activity --case bridge --sample --output outputs/sample_check
python run.py slabs --case bridge --sample --output outputs/sample_generation
```

The sample contains trial 0, composition 0, run 0, and all 18 temperatures. The annealing schedule and model are unchanged. Compare with `sample/expected_activity.json` and `sample/slabs/`. Full production slab and activity collections are not included due to their size. See the root README for the complete figure and postprocessing workflow.

## Equiatomic calculations

See [equi-atomic/README.md](equi-atomic/README.md) for the independent 20-run equiatomic CEMC calculations and Figures 3-6, S13-S15.
