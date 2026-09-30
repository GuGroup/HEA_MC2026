# Fig S10 reproduction

Run `python plot.py` from this folder, or invoke this script by its full path. Install the packages in the root `requirements.txt` first. This folder contains the numerical inputs and all plotting code required for this figure; no server paths are accessed. Outputs are written under `outputs/`.

- (a) Study 1: 1400 compositions
- (b) Study 2: ML1
- (c) Study 2: ML2
- (d) Study 2: ML3
- (e) Study 2: ML4
- (f) Study 2: ML5
- (g) Study 2: ML6
- (h) Study 3: fcc(100) hollow
- (i) Study 3: fcc(100) bridge
- (j) Study 3: fcc(110) bridge
- (k) Study 3: fcc(111) fcc
- (l) Study 3: fcc(111) top

The plotting data preserve the original selected trial IDs, experiment values, t-SNE or physical-grid coordinates, and scaling. The `analysis_data/` folder at the package root contains the complete trial-level metric tables and provenance needed to audit selections. `Fig_S10` writes two pages with six rows each.
