#!/usr/bin/env python3
"""Slab-run cluster bootstrap CIs for CEMC mass in Random bin support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_hbe_distributions_log_kde_peak as analysis


N_BOOTSTRAP = 10_000
N_RUNS = 20
CI_PERCENTILES = (2.5, 97.5)
HISTOGRAM_DIRNAME = (
    "hbe_distribution_analysis_log_activity_KDE_peak_trial_BE_shift_"
    "probability_mass_weighted_histogram_1x5"
)
OUTPUT_ROOT = analysis.BASE / "log_activity_KDE_peak_trial_paired_joint_bootstrap"
OUTPUT_PAIR = OUTPUT_ROOT / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_by_case_element.csv"
OUTPUT_CASE = OUTPUT_ROOT / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_by_case.csv"
OUTPUT_JSON = OUTPUT_ROOT / "CEMC_in_Random_probability_mass_slab_bootstrap_95CI_summary.json"
OUTPUT_NPZ = OUTPUT_ROOT / "CEMC_in_Random_probability_mass_slab_bootstrap_replicates.npz"


def load_exact_edges(workdir: Path) -> np.ndarray:
    path = workdir / HISTOGRAM_DIRNAME / "weighted_histogram_bin_data.csv"
    frame = pd.read_csv(path)
    first = frame[frame["element"] == analysis.ELEMENTS[0]].sort_values("bin_index")
    edges = np.r_[first["bin_left_eV"].to_numpy(float),
                  first["bin_right_eV"].to_numpy(float)[-1]]
    if len(edges) != 91:
        raise ValueError(f"Expected 91 edges in {path}, found {len(edges)}")
    return edges


def hist_by_run(values_by_run, fractions_by_run, element_index, edges):
    presence = np.zeros((N_RUNS, len(edges) - 1), dtype=bool)
    weight = np.zeros((N_RUNS, len(edges) - 1), dtype=float)
    for run in range(N_RUNS):
        values = values_by_run[run]
        fractions = fractions_by_run[run][:, element_index]
        mask = fractions > 0.0
        counts, _ = np.histogram(values[mask], bins=edges)
        weighted, _ = np.histogram(values[mask], bins=edges, weights=fractions[mask])
        presence[run] = counts > 0
        weight[run] = weighted
    return presence, weight


def bootstrap_containment(rs_presence, cemc_weight, sample_indices, fixed_support):
    joint = np.empty(len(sample_indices), dtype=float)
    conditional = np.empty(len(sample_indices), dtype=float)
    chunk_size = 500
    for start in range(0, len(sample_indices), chunk_size):
        stop = min(start + chunk_size, len(sample_indices))
        idx = sample_indices[start:stop]
        sampled_support = np.any(rs_presence[idx], axis=1)
        sampled_weight = np.sum(cemc_weight[idx], axis=1)
        denominator = np.sum(sampled_weight, axis=1)
        joint[start:stop] = (
            np.sum(sampled_weight * sampled_support, axis=1) / denominator)
        conditional[start:stop] = (
            np.sum(sampled_weight * fixed_support[None, :], axis=1) / denominator)
    return joint, conditional


def evaluate_case(facet, site, case):
    workdir = analysis.BASE / facet / site
    model = analysis.parse_model(workdir / "inputs/activity_model.txt")
    _, shifts, _ = analysis.load_shift_data(case)
    edges = load_exact_edges(workdir)
    random_values = [None] * N_RUNS
    random_fractions = [None] * N_RUNS
    cemc_values = [[] for _ in range(N_RUNS)]
    cemc_fractions = [[] for _ in range(N_RUNS)]

    with tempfile.TemporaryDirectory(prefix=f"bootstrap_{facet}_{site}_") as tmp:
        tmpdir = Path(tmp)
        for run in range(N_RUNS):
            result = analysis.evaluate_sites(
                model, analysis.reconstruct_random(workdir, run, tmpdir), shifts)
            random_values[run] = result[2] - model.e_opt
            random_fractions[run] = result[3]
        for temperature in analysis.TEMPERATURES:
            records = analysis.load_cemc_records(workdir, temperature)
            for run in range(N_RUNS):
                result = analysis.evaluate_sites(model, records[run], shifts)
                cemc_values[run].append(result[2] - model.e_opt)
                cemc_fractions[run].append(result[3])

    cemc_values = [np.concatenate(rows) for rows in cemc_values]
    cemc_fractions = [np.vstack(rows) for rows in cemc_fractions]
    rng = np.random.default_rng(
        analysis.stable_seed("CEMC-in-RS-slab-cluster-bootstrap-v1", case))
    sample_indices = rng.integers(0, N_RUNS, size=(N_BOOTSTRAP, N_RUNS))
    rows = []
    joint_by_element = []
    conditional_by_element = []
    for ei, element in enumerate(analysis.ELEMENTS):
        rs_presence, _ = hist_by_run(
            random_values, random_fractions, ei, edges)
        _, cemc_weight = hist_by_run(cemc_values, cemc_fractions, ei, edges)
        full_support = np.any(rs_presence, axis=0)
        full_weight = np.sum(cemc_weight, axis=0)
        estimate = float(np.sum(full_weight[full_support]) / np.sum(full_weight))
        joint, conditional = bootstrap_containment(
            rs_presence, cemc_weight, sample_indices, full_support)
        joint_by_element.append(joint)
        conditional_by_element.append(conditional)
        j_lo, j_hi = np.percentile(joint, CI_PERCENTILES)
        c_lo, c_hi = np.percentile(conditional, CI_PERCENTILES)
        rows.append({
            "facet": facet, "site": site, "case": case, "element": element,
            "n_runs": N_RUNS, "n_temperatures": len(analysis.TEMPERATURES),
            "n_bootstrap": N_BOOTSTRAP, "bootstrap_unit": "paired slab run cluster",
            "point_estimate_percent": 100.0 * estimate,
            "paired_joint_bootstrap_mean_percent": 100.0 * float(np.mean(joint)),
            "paired_joint_95CI_lower_percent": 100.0 * float(j_lo),
            "paired_joint_95CI_upper_percent": 100.0 * float(j_hi),
            "paired_joint_95CI_lower_at_least_90_percent": bool(j_lo >= 0.90),
            "conditional_fixed_RS_bootstrap_mean_percent": 100.0 * float(np.mean(conditional)),
            "conditional_fixed_RS_95CI_lower_percent": 100.0 * float(c_lo),
            "conditional_fixed_RS_95CI_upper_percent": 100.0 * float(c_hi),
            "conditional_fixed_RS_95CI_lower_at_least_90_percent": bool(c_lo >= 0.90),
        })
    return rows, np.vstack(joint_by_element), np.vstack(conditional_by_element)


def summarize_group(label, point, joint, conditional):
    joint_mean = np.mean(joint, axis=0)
    conditional_mean = np.mean(conditional, axis=0)
    j_lo, j_hi = np.percentile(joint_mean, CI_PERCENTILES)
    c_lo, c_hi = np.percentile(conditional_mean, CI_PERCENTILES)
    return {
        "case": label,
        "n_elements": len(point),
        "point_macro_mean_percent": 100.0 * float(np.mean(point)),
        "paired_joint_bootstrap_macro_mean_percent": 100.0 * float(np.mean(joint_mean)),
        "paired_joint_95CI_lower_percent": 100.0 * float(j_lo),
        "paired_joint_95CI_upper_percent": 100.0 * float(j_hi),
        "paired_joint_95CI_lower_at_least_90_percent": bool(j_lo >= 0.90),
        "conditional_fixed_RS_bootstrap_macro_mean_percent": 100.0 * float(np.mean(conditional_mean)),
        "conditional_fixed_RS_95CI_lower_percent": 100.0 * float(c_lo),
        "conditional_fixed_RS_95CI_upper_percent": 100.0 * float(c_hi),
        "conditional_fixed_RS_95CI_lower_at_least_90_percent": bool(c_lo >= 0.90),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    case_rows = []
    all_joint = []
    all_conditional = []
    for facet, site, case in analysis.COMBOS:
        rows, joint, conditional = evaluate_case(facet, site, case)
        pair_rows.extend(rows)
        point = np.asarray([row["point_estimate_percent"] / 100.0 for row in rows])
        case_rows.append(summarize_group(case, point, joint, conditional))
        all_joint.append(joint)
        all_conditional.append(conditional)
        print(f"Ready {case}", flush=True)
    pair_frame = pd.DataFrame(pair_rows)
    case_frame = pd.DataFrame(case_rows)
    pair_frame.to_csv(OUTPUT_PAIR, index=False)
    case_frame.to_csv(OUTPUT_CASE, index=False)
    all_joint = np.vstack(all_joint)
    all_conditional = np.vstack(all_conditional)
    case_conditional = np.vstack([
        np.mean(all_conditional[i * len(analysis.ELEMENTS):(i + 1) * len(analysis.ELEMENTS)], axis=0)
        for i in range(len(analysis.COMBOS))
    ])
    case_joint = np.vstack([
        np.mean(all_joint[i * len(analysis.ELEMENTS):(i + 1) * len(analysis.ELEMENTS)], axis=0)
        for i in range(len(analysis.COMBOS))
    ])
    overall_conditional = np.mean(all_conditional, axis=0)
    overall_joint = np.mean(all_joint, axis=0)
    np.savez_compressed(
        OUTPUT_NPZ,
        pair_joint_percent=100.0 * all_joint,
        case_joint_percent=100.0 * case_joint,
        overall_joint_percent=100.0 * overall_joint,
        pair_conditional_percent=100.0 * all_conditional,
        case_conditional_percent=100.0 * case_conditional,
        overall_conditional_percent=100.0 * overall_conditional,
        pair_case=np.asarray(pair_frame["case"], dtype="U32"),
        pair_element=np.asarray(pair_frame["element"], dtype="U8"),
        cases=np.asarray(case_frame["case"], dtype="U32"),
    )
    all_point = pair_frame["point_estimate_percent"].to_numpy(float) / 100.0
    overall = summarize_group("all_25_case_element_pairs", all_point,
                              all_joint, all_conditional)
    summary = {
        "method": "10,000-replicate paired slab-run cluster percentile bootstrap",
        "resampling": (
            "Sample 20 run IDs with replacement; use the same sampled IDs for Random "
            "and CEMC; retain all temperatures and adsorption sites within each run."
        ),
        "paired_joint_CI": "Recompute Random occupied-bin support and CEMC mass in each replicate.",
        "conditional_fixed_RS_CI": "Keep full-sample Random support fixed; resample CEMC slab runs.",
        "caution": (
            "Occupied-support estimation is discontinuous; joint bootstrap replicates contain "
            "fewer unique Random slabs on average and therefore tend to shrink empirical support."
        ),
        "overall": overall,
        "n_pair_CI_lower_at_least_90_joint": int(
            pair_frame["paired_joint_95CI_lower_at_least_90_percent"].sum()),
        "n_pair_CI_lower_at_least_90_conditional": int(
            pair_frame["conditional_fixed_RS_95CI_lower_at_least_90_percent"].sum()),
        "outputs": {
            "by_case_element_csv": str(OUTPUT_PAIR),
            "by_case_csv": str(OUTPUT_CASE),
            "conditional_replicates_npz": str(OUTPUT_NPZ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
