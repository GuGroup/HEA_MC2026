#!/usr/bin/env python3
"""Select one high-tau/low-MSE/low-CRPS trial from saved scores and plot its map."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_tau_mse_crps_modal_activity_map_kde import (
    high_tau_low_errors_selection,
    load_activity_for_trial,
    load_matched_activity,
    load_mc_activity_from_zarr,
    load_score_points,
    plot_experiment_mc_map,
    scale_prediction_to_minus1_0,
    to_grid,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--world-size", type=int, default=5)
    p.add_argument("--best-score-dir", default=None)
    p.add_argument("--activity-scale", choices=["scaled", "raw"], default="scaled")
    p.add_argument("--log-transform", action="store_true")
    p.add_argument("--trial-min", type=int, default=None)
    p.add_argument("--trial-max", type=int, default=None)
    p.add_argument("--mask", default="/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy")
    p.add_argument("--experimental-nolog", default="orr_matched_activity.json")
    p.add_argument("--experimental-log", default="orr_log_matched_activity.json")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cmap", default="bwr_r")
    p.add_argument("--dpi", type=int, default=250)
    args = p.parse_args()

    root = Path(args.results_root).resolve()
    domain = "log" if args.log_transform else "nolog"
    score_dir = Path(args.best_score_dir).resolve() if args.best_score_dir else (
        root / f"crps_selection_activity_{domain}"
    )
    output = Path(args.output_dir).resolve() if args.output_dir else (
        root / f"high_tau_low_mse_crps_activity_map_{args.activity_scale}_{domain}"
    )
    output.mkdir(parents=True, exist_ok=True)

    cemc, _ = load_score_points(
        score_dir, args.trial_min, args.trial_max,
        apply_log=False, cemc_candidates="best-temperature",
    )
    selected = high_tau_low_errors_selection(cemc)
    trial, temperature = int(selected["trial"]), int(selected["temperature"])

    activity = load_activity_for_trial(root, trial, args.world_size)
    shard = str(activity["source_shard"].iloc[0])
    log_values = load_mc_activity_from_zarr(root, shard, trial, temperature, "raw")
    predicted_unscaled = log_values if args.log_transform else np.exp(log_values)
    predicted = scale_prediction_to_minus1_0(predicted_unscaled)

    mask = np.load(args.mask).astype(bool)
    n_valid = int(np.count_nonzero(~mask))
    exp_path = Path(args.experimental_log if args.log_transform else args.experimental_nolog).resolve()
    experimental = load_matched_activity(exp_path, n_valid)
    exp_grid, predicted_grid = to_grid(experimental, mask), to_grid(predicted, mask)

    png = output / f"high_tau_low_mse_crps_trial_{trial:04d}_T_{temperature:04d}_activity_map.png"
    plot_experiment_mc_map(
        exp_grid, predicted_grid, selected, png,
        args.log_transform, args.cmap, args.dpi,
    )
    record = {
        "mode": domain, "activity_scale": args.activity_scale,
        "trial": trial, "temperature": temperature,
        "tau": float(selected["tau"]), "mse": float(selected["mse"]),
        "crps": float(selected["crps"]),
        "ideal_d_tau": float(selected["_ideal_d_tau"]),
        "ideal_d_mse": float(selected["_ideal_d_mse"]),
        "ideal_d_crps": float(selected["_ideal_d_crps"]),
        "ideal_score": float(selected["_ideal_score"]), "source_shard": shard,
    }
    pd.DataFrame([record]).to_csv(output / "high_tau_low_mse_crps_representative_trial.csv", index=False)
    pd.DataFrame({"composition_index": np.arange(n_valid), "experimental_activity": experimental,
                  "selected_mc_activity": predicted}).to_csv(
        output / f"high_tau_low_mse_crps_trial_{trial:04d}_activity_values.csv", index=False
    )
    np.save(output / f"high_tau_low_mse_crps_trial_{trial:04d}_grid.npy", predicted_grid)
    print(
        f"selected trial={trial}, T={temperature} K, tau={selected['tau']:.6f}, "
        f"mse={selected['mse']:.6f}, crps={selected['crps']:.6f}, "
        f"ideal_score={selected['_ideal_score']:.6f}"
    )
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
