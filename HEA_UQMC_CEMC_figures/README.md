# HEA UQMC and CEMC Figure Reproduction Package

This package reproduces Table 1, Figs. 1-6 and Figs. S6-S15 from the supplied Manuscript and Supporting Information. It includes numerical plotting data, portable plotting code, the original C++ simulation backends and model inputs, commands for generating and reusing CEMC slabs, and validation records.

## Reproduce Table 1

```bash
python Table_1/reproduce.py
```

`Table_1/` is a standalone folder containing paired scores for 10,000 trials per dataset (12 datasets), the calculation code, a manuscript reference, and generated CSV, Markdown and HTML tables. Only NumPy is required. See `Table_1/README.md` for definitions and verification details. All 52 displayed probabilities match Table 1. Two percentile endpoints differ from the supplied manuscript by 0.01; the calculated values and manuscript values are both documented, without modifying the input data. The figure-only launcher does not run the table script.

## Quick start: reproduce the figures

Use Python 3.14 with the validated package versions, or a compatible Python environment:

```bash
conda create -n hea-figures python=3.14 -y
conda activate hea-figures
cd /path/to/HEA_UQMC_CEMC_figures
python -m pip install -r requirements.txt
python reproduce_all_figures.py
```

To reproduce only one figure:

```bash
python Fig_1/plot.py
python Fig_S12/plot.py
```

Each figure folder contains `plot.py`, `figure.json`, `data/`, generated `outputs/`, and any required local plotting modules. Its scripts resolve paths from their own location, so they also work when invoked from another current directory. Each folder can be used independently with the listed Python dependencies. Individual panels retain the original numerical values; combined figures add panel labels and dataset names. Fig. S10 is split into two pages with six rows each.

| Folder | Contents |
|---|---|
| `Table_1` | Paired trial scores, improvement probabilities and percentile intervals for all 12 datasets |
| `Fig_1` | Study 1; Study 2 ML4; Study 3 fcc(100) hollow: probability-colored 3D metric plots |
| `Fig_2` | The same three datasets: most-probable-trial activity maps |
| `Fig_3` | Equiatomic surface composition and Warren-Cowley order |
| `Fig_4` | Equiatomic BE densities: homogeneous, CEMC and layer-shuffled CEMC |
| `Fig_5` | Homogeneous BE distributions with activity volcanoes |
| `Fig_6` | Conditional BE probability masses and containment |
| `Fig_S6` | Study 2 ML1, ML2, ML3, ML5, ML6: 3D metric plots |
| `Fig_S7` | Study 3 fcc(100) bridge, fcc(110) long bridge, fcc(111) hollow, fcc(111) top: 3D metric plots |
| `Fig_S8` | Study 2 ML1, ML2, ML3, ML5, ML6: most-probable activity maps |
| `Fig_S9` | The four remaining Study 3 sites: most-probable activity maps |
| `Fig_S10` | Best-performing-trial activity maps for Study 1, all six MLs, and all five HER sites |
| `Fig_S11` | Continuous recall for Study 2 ML1-ML6 |
| `Fig_S12` | Continuous recall for all five HER sites and Study 1 |

## Directory layout

```text
HEA_UQMC_CEMC_figures/
  README.md
  requirements.txt
  reproduce_all_figures.py
  Fig_1/, Fig_2/, Fig_S6/ ... Fig_S12/
  workflows/
    PtPdRhRuIr_fcc111/           # Study 2 ML1-ML6
      Study1/                   # Study 1, same physical facet, original backend/format
    FeCoNiPdPt_fcc111/           # hollow and top share one slab trajectory set
    FeCoNiPdPt_fcc110/           # long bridge
    FeCoNiPdPt_fcc100/           # hollow and bridge share one slab trajectory set
  analysis/                     # Metrics and selection from newly evaluated activities
  analysis_data/                # Full metric tables, selected values, and selection records
  validation/                   # Numerical, sample simulation, and image checks
  provenance.json
  manifest_sha256.json
```

## Generate CEMC slabs once per facet and composition dataset

Simulation requires Linux, a C++17 compiler, and MPI (`mpicxx` and `mpirun` on PATH). Python dependencies alone are sufficient for plotting. The archive contains source code rather than machine-specific compiled executables.

The four physical facet models are PtPdRhRuIr fcc(111), FeCoNiPdPt fcc(111), fcc(110), and fcc(100). The supplied CE exports include the lattice topology and interaction coefficients, so there is no need to retrain CE models before running these simulations.

Example for FeCoNiPdPt fcc(100):

```bash
cd workflows/FeCoNiPdPt_fcc100
python run.py build
python run.py slabs --case hollow --ranks 48
python run.py activity --case hollow --ranks 48
python run.py activity --case bridge --ranks 48
python run.py random --case hollow --ranks 48
python run.py random --case bridge --ranks 48
```

The first command after building creates `outputs/slabs_shared_facet/`. Both activity commands read that same slab set and use their respective adsorption models. They do not repeat CEMC. Random/homogeneous structures are reconstructed deterministically from their seeds.

The analogous cases are:

| Workflow | Cases | Slab reuse |
|---|---|---|
| `FeCoNiPdPt_fcc111` | `fcc`, `top` | `fcc` means fcc hollow; both share one facet slab set |
| `FeCoNiPdPt_fcc110` | `bridge` | Long-bridge activity model |
| `FeCoNiPdPt_fcc100` | `hollow`, `bridge` | Both share one facet slab set |
| `PtPdRhRuIr_fcc111` | `ML1` through `ML6` | Separate composition datasets, one common physical facet |

For example:

```bash
cd /path/to/HEA_UQMC_CEMC_figures/workflows/PtPdRhRuIr_fcc111
python run.py build
python run.py slabs --case ML4 --ranks 48
python run.py activity --case ML4 --ranks 48
python run.py random --case ML4 --ranks 48
```

MLs and Study 1 contain different composition lists. Their trajectories cannot be interchanged merely because the physical facet is the same. Site models on an identical facet/composition/seed/schedule dataset can reuse the same trajectories. Different facets, compositions, schedules, CE coefficients, or composition uncertainty settings require their own trajectories.

Original production settings are 10,000 trials, 20 runs per composition, 1,000 metal atoms per slab, and 18 snapshots from 2,000 to 300 K. The annealing schedule itself goes from 2,023 to 298 K. Original random seeds and composition/BE uncertainty settings are retained in `configs/*.json`. Paths are made absolute only in generated runtime configurations under `outputs/runtime_configs/`.

## Reuse existing slabs or change an adsorption model

```bash
python run.py activity --case hollow --ranks 48 \
  --slab-dir /path/to/existing/cemc_slabs_3bit \
  --output outputs/hollow_new_model \
  --activity-model /path/to/new_exported_activity_model.txt \
  --be-mean 0.0 --be-sigma 0.1935
```

The model must use the exported text format and site topology of the corresponding backend; a training-model JSON cannot be substituted directly. The supplied model exports in `inputs/` provide working examples. Changing coefficients or the BE shift preserves the slab trajectories. Changing site geometry requires a compatible site-topology export. The energy convention is `E_used = E_model - BE_shift`.

For FeCoNiPdPt, the bundled production evaluation uses the newer H-BE shifts (mean 0.0 eV, sigma 0.1935 eV). For PtPdRhRuIr, the supplied OH shifts have mean 0.04233 eV and sigma 0.2604 eV. Original element orders are retained and must not be rearranged.

Slab generation supports `--trial-start` and `--trial-end` (exclusive). Use complete 40-trial shard boundaries for resumable production runs. Completed generation shards are skipped. Activity evaluation should use an empty output directory; the wrapper rejects a nonempty directory because some original summaries append rows. Use `--output` to keep each model evaluation separate.

## Small runnable verification example

Every main workflow includes actual packed slabs from trial 0, composition 0, run 0, at all 18 temperatures:

```bash
python run.py build
python run.py activity --case hollow --sample --output outputs/sample_hollow
python run.py slabs --case hollow --sample --output outputs/sample_generated_slabs
python inspect_slab3bit.py sample/slabs/trials_00000_00000.bin
```

Replace `hollow` with a valid case for that workflow. The Pt sample is ML1 only. These commands exercise the real model and annealing schedule with a small dataset. They are not full production calculations. Reference sample activities and provenance are under `sample/`.

## Postprocess newly evaluated Study 2/3 activities

From the package root:

```bash
python analysis/postprocess_activity.py \
  --dataset Fe_fcc100_hollow \
  --cemc workflows/FeCoNiPdPt_fcc100/outputs/hollow_activity/activity_shards \
  --random workflows/FeCoNiPdPt_fcc100/outputs/hollow_random/random_activity_shards \
  --output new_analysis/Fe_fcc100_hollow
```

Dataset names are listed in `cases.json`: `Pt_ML1` through `Pt_ML6`, `Fe_fcc100_hollow`, `Fe_fcc100_bridge`, `Fe_fcc110_bridge`, `Fe_fcc111_fcc`, and `Fe_fcc111_top`. The code takes the natural logarithm of each slab activity, averages over runs, scales each composition map independently to [-1, 0], and computes the original metrics. It uses the saved experimental values and map coordinates for that composition dataset. It is not intended for an unrelated composition list without supplying the corresponding experimental/layout data.

The postprocessor writes full-temperature metric tables, one selected temperature per trial, histogram probabilities, most-probable/best trial records, and activity tables. A `mean_log_cache/` is written for selecting maps without rereading raw shards. The bundled figure folders reproduce the historical results from frozen plotting inputs; running a new model does not silently replace those inputs.

Study 2's historical homogeneous metrics were calculated from double-precision log means. The portable general evaluator stores individual activities as float32, matching the CEMC sidecars and Study 3. Fresh Study 2 homogeneous metrics can therefore differ slightly through rounding. The exact historical plotting tables are included, as is the original double-precision homogeneous log-mean generator in the Pt workflow.

## Study 1 execution

See `workflows/PtPdRhRuIr_fcc111/Study1/README.md`. Study 1 preserves its original packed trajectory format, original composition inputs, and separate activity/CRPS code. It belongs under the same Pt fcc(111) physical facet but is a separate experimental dataset and original implementation.

## Numerical conventions retained from the original figures

- Activity maps use -1 for the most active and 0 for the least active. Recall uses ascending stable activity ranks through the top 60%.
- Study 2 and Study 3 select one CEMC temperature per trial using equal-weight range-normalized distance to maximum Kendall tau and minimum MSE/CRPS. Their figure probability colors use a 20 x 20 x 20 histogram smoothed with Gaussian sigma 1 bin.
- Study 2 most-probable representatives require positive tau. Study 3 representatives use the global highest-probability bin without that constraint. Best-performing trials minimize the original normalized distance criterion.
- Study 1 originally selected its representative with continuous Gaussian KDE, whereas its final 3D color values use the smoothed histogram. Its best CEMC trial is chosen from all trial/temperature pairs. Those distinct rules are preserved.
- Study 1 uses Gaussian CRPS with run standard deviation. Studies 2/3 use the original empirical-distribution discrepancy called CRPS in their scripts. These metrics have not been replaced with a single common formula.
- Study 1 maps retain 1st/99th percentile clipping. Its 3D metric tables and maps contain 1,400 compositions. Its original continuous-recall code excludes the central grid row and column, leaving 1,317 compositions, and reapplies clipping on that subset. The different counts are intentional reproductions of the source code.
- Fe activity maps retain saved t-SNE coordinates. Recomputing t-SNE with another library version can change layout even when activities are unchanged.

## Archive scope and storage

The multi-terabyte production slab collections and raw activity collections are not copied. One Study 2 ML slab set alone occupies about 462 GB; one Fe facet set occupies about 321 GB before archive compression. The package includes all figure inputs, original simulation inputs/model exports, source code, and small actual slab samples. Thus all figures can be reproduced without regenerating slabs. Full simulations require adequate storage and substantial compute time.

Credentials, host login settings, job logs, caches, and compiled binaries are not part of the archive. Source server paths appear only as provenance. No login is required to reproduce the figures or run the supplied simulations locally.

## Validation

The `validation/` records document the checks performed while packaging:

- All nine figure scripts ran successfully, producing every requested panel.
- All 11 Study 2/3 datasets reproduce the original temperature selection, most-probable and best trial IDs, and histogram probabilities from their full metric tables.
- The 11 Study 2/3 histogram panels are pixel-identical to the source PNGs in the validated environment. Study 1 preserves identical plotted inputs, with about 98.8% identical image pixels; rendering differences remain.
- All continuous-recall arrays match their original computation/reference values.
- For all four main workflows, regenerated sample slab occupations match the originals byte for byte. All six sampled adsorption-model evaluations match the source float32 activities exactly.
- Study 1 also has a packed-slab regeneration and log-activity comparison check.
- A full 40-trial Study 3 activity shard was processed and its metric values compared with the original tables.

These checks do not claim that the entire multi-terabyte, 10,000-trial production simulation was rerun. Different compiler, standard-library, MPI, or Python versions may produce small numerical or rendering differences. `manifest_sha256.json` verifies the delivered files.

## Equiatomic workflows and figure details

The archive now includes all 16 requested figures: 1-6 and S6-S15. New figure folders are `Fig_3`, `Fig_4`, `Fig_5`, `Fig_6`, `Fig_S13`, `Fig_S14`, and `Fig_S15`. Each has a portable `plot.py`, numeric data, an English README and regenerated PNG/PDF outputs.

```bash
pip install -r requirements.txt
python reproduce_all_figures.py
# Fast single-figure reproduction:
python Fig_3/plot.py
# Recompute equiatomic numerical inputs from packed atomistic structures:
python analysis/equiatomic/recompute.py
```

| Figure | Contents |
|---|---|
| 3 | Surface composition and Warren-Cowley curves, Pt fcc111 and Fe fcc100 |
| 4 | Homogeneous/CEMC/layer-shuffled BE densities, Pt fcc111 and Fe fcc100 hollow |
| 5 | Homogeneous BE distributions and activity volcanoes |
| 6 | Conditional BE mass and containment, Pt fcc111 and Fe fcc100 hollow |
| S13 | Final layer-pair compositions for all four facets |
| S14 | Zero-shift H-BE densities for the other four Fe facet/site cases |
| S15 | Shifted conditional H-BE mass and containment for those cases |

An independent `equi-atomic/` folder is supplied under each of:

- `workflows/PtPdRhRuIr_fcc111/`
- `workflows/FeCoNiPdPt_fcc111/`
- `workflows/FeCoNiPdPt_fcc110/`
- `workflows/FeCoNiPdPt_fcc100/`

Their English READMEs describe compilation, full 20-run CEMC generation, 10 K snapshot generation, ASE/CIF export, reuse of structures with another adsorption model, and exact BE shifts. Complete equiatomic ensembles are small enough to include: six historical site cases each contain 20 homogeneous slabs and 380 CEMC snapshots; four additional dense archives contain 3,440 snapshots each. Geometry is stored once per ensemble and atom identities are compressed losslessly. The original dense archive coordinate precision is retained.

For exact historical figure reproduction, retain the site-specific seeds supplied in the configurations. For new predictions, one CEMC ensemble per physical facet can be reused with compatible site models via `--structures`. See `analysis/equiatomic/README.md` for statistical conventions and validation. All original pre-extension model/data files are preserved; only shared README/requirements/figure-launcher/manifest files are updated.

Validation: four C++ backends compiled, one complete 19-temperature trajectory per facet matched the original atomic numbers exactly, and all supplied equiatomic numeric figure tables agreed with recalculation within floating-point precision. Composite plots are rerendered from numbers and visually checked; typography and spacing are not guaranteed pixel-identical to the Word images. The Fig. 4 Fe layer-shuffled activity is 0.1621249547, matching the supplied figure (0.16).
