# Study 1 — Pt–Pd–Ir–Rh–Ru composition and ORR activity

This directory pairs reconstructed alloy compositions with digitized oxygen reduction reaction (ORR) currents at 1,400 positions. The elemental maps and the ORR map come from the supporting-information figure archived as `adma202207635-sup-0001-suppmat.png`.

> Banko, L. *et al.* Microscale Combinatorial Libraries for the Discovery of High‐Entropy Materials. *Adv. Mater.* **35**, 2207635 (2023). [DOI](https://doi.org/10.1002/adma.202207635)

## Composition

`new_get_pixels.py` resizes the five `*_padding.png` elemental maps to a common 51 × 51 grid. For each element, it reconstructs a density map by summing rotated two-dimensional Gaussian peaks with the centers, widths, and angles specified in the script. It compares each reconstructed map with the corresponding image and divides each element’s density by the sum of all five densities at the same grid position to estimate its atomic fraction.

The CSV export code converts each position to a 512-atom composition. It rounds each element’s atom count down, assigns the remaining atoms to Pt, and divides the final counts by 512. `composition.csv` and `composition_new.csv` are identical. Each row contains the zero-based `Pixel_Index`, atomic fractions in the order `Ru, Rh, Pd, Ir, Pt`, and their sum (`Total`).

## ORR activity

`get_exp_activities.py` resizes `ORR.png` to the same 51 × 51 grid using area averaging. At each position it subtracts the blue channel from the red channel. Using only positions where `static_grid_mask.npy` is false, it maps the smallest channel difference (blue) to −11 pA and the largest (red) to −25 pA, interpolating linearly to estimate the current between these endpoints.

The directory contains two dimensionless activity scales. In the formulas below, $I$ is the numerical current in pA. Both scales assign **0** to −11 pA and **−1** to −25 pA.

| Scale | Activity | Full grid | Valid positions keyed by `Pixel_Index` |
| --- | --- | --- | --- |
| Linear | $(I+11)/14$ | `orr_scaled_activity.npy` | `orr_matched_activity.json` |
| Logarithmic | $\ln(I+26)/\ln(15)-1$ | `orr_log_scaled_activity.npy` | `orr_log_matched_activity.json` |

To pair activity with composition, look up each CSV row’s `Pixel_Index` as a string key in the JSON for the desired scale. Indices 0–1399 enumerate valid grid positions from left to right, then top to bottom. Apply the same mask when reading the full-grid arrays.

To rerun extraction from the supplied crops, set the local input paths in both scripts to this directory. The composition CSV export block in `new_get_pixels.py` is commented out. The active branch of `get_exp_activities.py` writes the logarithmic outputs. To regenerate the linear files, select the commented linear scaling block in place of the logarithmic calculation and set both output names to the linear filenames above.