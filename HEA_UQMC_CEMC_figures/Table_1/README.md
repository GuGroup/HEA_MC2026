# Table 1: probability of improvement with CEMC

This standalone folder recomputes Table 1 from the original, paired trial-level scores for Study 1, Study 2 ML1-ML6, and five facet/site cases in Study 3. Each dataset contains 10,000 trials (IDs 0-9999). You do not need to regenerate slabs or rerun UQMC to reproduce this table.

## Run

Use Python 3.10 or newer with NumPy:

```bash
python -m pip install -r Table_1/requirements.txt
python Table_1/reproduce.py
```

Run these commands from the extracted package root. Alternatively, copy this entire folder anywhere and run `python /path/to/Table_1/reproduce.py`; paths are resolved relative to the script. Use `--output-dir /path/to/results` to select another output directory.

## Inputs and calculation

`data/*.csv.gz` contains the CEMC and homogeneous Kendall tau-b, MSE and CRPS scores, paired by the original trial ID, plus the selected CEMC temperature and number of compositions. Decimal score strings are copied directly from the existing `analysis_data` CSVs without rounding. `data/provenance.json` records source paths and SHA-256 checksums. Source paths are provenance only and are not needed at runtime.

The differences are:

- delta tau = tau(CEMC) - tau(homogeneous).
- delta MSE = MSE(homogeneous) - MSE(CEMC).
- delta CRPS = CRPS(homogeneous) - CRPS(CEMC).

A positive difference indicates better agreement with experiment for CEMC. Each probability is the count of strictly positive differences divided by 10,000. Ties are not improvements. The joint probability counts trials where all three differences are positive. Every trial receives equal weight; the KDE probabilities used to display figures are not weights here.

The bracketed intervals are the 2.5th and 97.5th percentiles of the trial differences, calculated with NumPy's linear interpolation. They describe the distribution of differences, not uncertainty intervals on the probabilities. The Average row is the arithmetic mean of the 12 unrounded dataset probabilities. Its intervals are intentionally blank.

These frozen scores preserve the original study-specific scoring and temperature selection. Study 1 uses the original Gaussian CRPS calculation; Studies 2 and 3 use the original empirical distribution discrepancy called CRPS in the source analysis. The table script does not replace those definitions or select a different temperature for each metric. The full package includes the upstream workflow and analysis code for regenerating activities and scores.

## Outputs and verification

- `outputs/table1.csv`: formatted table, two decimal places.
- `outputs/table1_full_precision.csv`: numeric probabilities and percentile endpoints before display rounding.
- `outputs/table1.md` and `outputs/table1.html`: readable tables.
- `outputs/validation.json`: trial counts, improvement counts, ties, and cell-by-cell comparison with the manuscript.
- `manuscript_reference.json`: values extracted from Table 1 of the supplied Manuscript.docx, including its checksum. These values are used only for comparison, never to generate the calculated table.

All 52 displayed probabilities (12 dataset rows plus Average, four probabilities each) match the manuscript. Two interval endpoints do not match at two decimals:

| Dataset and endpoint | Manuscript | Recomputed |
|---|---:|---:|
| Study 1, delta tau 97.5th percentile | 0.39 | 0.38 |
| Study 2 ML4, delta CRPS 2.5th percentile | -0.04 | -0.03 |

The unrounded values are approximately 0.38378244 and -0.03494169. The supplied scores and the stated percentile definition do not reproduce these two manuscript endpoints; their cause has not been established. The script retains the calculated values rather than substituting the manuscript values. It reports discrepancies without treating known manuscript differences as a runtime failure.

The manuscript prose also assigns a CRPS improvement probability of 0.38 to ML5. Its Table 1 and the supplied data give ML5 = 0.46 and ML6 = 0.38. This package follows the calculated data and records this discrepancy explicitly.
