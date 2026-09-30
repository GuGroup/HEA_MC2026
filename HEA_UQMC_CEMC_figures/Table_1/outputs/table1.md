# Table 1: CEMC improvement probabilities

Intervals are the 2.5th and 97.5th percentiles of paired trial differences, not confidence intervals for the improvement probabilities. Average gives equal weight to the 12 datasets. Values are recomputed from the supplied scores; manuscript discrepancies are listed below.

| Dataset | P(delta tau > 0) [2.5%, 97.5%] | P(delta MSE > 0) [2.5%, 97.5%] | P(delta CRPS > 0) [2.5%, 97.5%] | P(all three > 0) |
|---|---|---|---|---|
| Study 1 | 0.46 [-0.66, 0.38] | 0.50 [-0.17, 0.23] | 0.51 [-0.16, 0.27] | 0.32 |
| Study 2, ML1 | 0.58 [-0.30, 0.89] | 0.63 [-0.07, 0.14] | 0.64 [-0.04, 0.05] | 0.40 |
| Study 2, ML2 | 0.78 [-0.22, 0.54] | 0.55 [-0.07, 0.06] | 0.36 [-0.06, 0.04] | 0.26 |
| Study 2, ML3 | 0.67 [-0.45, 0.68] | 0.54 [-0.09, 0.08] | 0.26 [-0.09, 0.03] | 0.18 |
| Study 2, ML4 | 0.55 [-0.72, 1.01] | 0.49 [-0.13, 0.14] | 0.50 [-0.03, 0.04] | 0.22 |
| Study 2, ML5 | 0.56 [-0.18, 0.25] | 0.59 [-0.05, 0.06] | 0.46 [-0.05, 0.04] | 0.17 |
| Study 2, ML6 | 0.71 [-0.29, 0.80] | 0.64 [-0.06, 0.12] | 0.38 [-0.05, 0.03] | 0.29 |
| Study 3, fcc(100) hollow | 0.57 [-0.36, 0.54] | 0.69 [-0.05, 0.08] | 0.68 [-0.06, 0.16] | 0.36 |
| Study 3, fcc(100) bridge | 0.60 [-0.40, 0.47] | 0.53 [-0.06, 0.07] | 0.53 [-0.08, 0.13] | 0.28 |
| Study 3, fcc(110) bridge | 0.61 [-0.27, 0.37] | 0.58 [-0.04, 0.07] | 0.54 [-0.06, 0.14] | 0.30 |
| Study 3, fcc(111) hollow | 0.44 [-0.61, 0.53] | 0.45 [-0.08, 0.08] | 0.53 [-0.10, 0.13] | 0.23 |
| Study 3, fcc(111) top | 0.65 [-0.24, 0.38] | 0.60 [-0.06, 0.09] | 0.68 [-0.07, 0.17] | 0.44 |
| Average | 0.60 | 0.57 | 0.51 | 0.29 |

## Comparison with the supplied manuscript

- Study 1, `tau_q975`: manuscript 0.39; recomputed 0.38.
- Study 2, ML4, `crps_q025`: manuscript -0.04; recomputed -0.03.

All improvement probabilities, including the Average row, match the manuscript at two decimal places.
