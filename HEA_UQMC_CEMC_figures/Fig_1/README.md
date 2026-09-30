# Fig 1 reproduction

Run `python plot.py` from this folder, or invoke this script by its full path. Install the packages in the root `requirements.txt` first. This folder contains the numerical inputs and all plotting code required for this figure; no server paths are accessed. Outputs are written under `outputs/`.

- (a) Study 1: 1400 compositions
- (b) Study 2: ML4
- (c) Study 3: fcc(100) hollow

The plotting data preserve the original selected trial IDs, experiment values, t-SNE or physical-grid coordinates, and scaling. The `analysis_data/` folder at the package root contains the complete trial-level metric tables and provenance needed to audit selections. `Fig_S10` writes two pages with six rows each.
