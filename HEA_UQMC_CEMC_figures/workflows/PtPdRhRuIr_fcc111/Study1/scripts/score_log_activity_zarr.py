#!/usr/bin/env python3
"""Score packed-activity Zarr outputs in the log-activity domain.

Creates the score tables expected by
plot_tau_mse_crps_modal_activity_map_kde.py without materializing the very
large all-snapshots activity CSV used by the legacy workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import kendalltau


def read_zarr_array(store: Path, name: str) -> np.ndarray:
    array_dir = store / name
    with (array_dir / ".zarray").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    shape = tuple(int(x) for x in metadata["shape"])
    chunks = tuple(int(x) for x in metadata["chunks"])
    if chunks != shape:
        raise ValueError(f"chunked arrays are not supported: {array_dir} has chunks={chunks}, shape={shape}")
    chunk = array_dir / ".".join("0" for _ in shape)
    values = np.fromfile(chunk, dtype=np.dtype(metadata["dtype"]))
    if values.size != int(np.prod(shape)):
        raise ValueError(f"bad chunk size in {chunk}: expected {np.prod(shape)}, got {values.size}")
    return values.reshape(shape).astype(np.float64)


def scale_log_prediction(mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(mean) & np.isfinite(sd)
    if not finite.all():
        raise ValueError("prediction contains non-finite mean or SD")
    span = float(np.max(mean) - np.min(mean))
    if span < 1.0e-30:
        return np.zeros_like(mean), np.zeros_like(sd)
    return -(mean - np.min(mean)) / span, np.abs(sd) / span


def gaussian_crps(mean: np.ndarray, sd: np.ndarray, observation: np.ndarray) -> float:
    sigma = np.abs(sd)
    safe = np.where(sigma > 1.0e-14, sigma, 1.0)
    z = (observation - mean) / safe
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    values = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    values = np.where(sigma > 1.0e-14, values, np.abs(observation - mean))
    return float(np.mean(values))


def metrics(mean: np.ndarray, sd: np.ndarray, experimental: np.ndarray) -> tuple[float, float, float]:
    if mean.shape != experimental.shape or sd.shape != experimental.shape:
        raise ValueError(
            f"composition shape mismatch: mean={mean.shape}, sd={sd.shape}, experimental={experimental.shape}"
        )
    prediction, prediction_sd = scale_log_prediction(mean, sd)
    tau = float(kendalltau(experimental, prediction, nan_policy="raise").statistic)
    mse = float(np.mean((experimental - prediction) ** 2))
    crps = gaussian_crps(prediction, prediction_sd, experimental)
    if not np.isfinite((tau, mse, crps)).all():
        raise ValueError(f"non-finite metric: tau={tau}, mse={mse}, crps={crps}")
    return tau, mse, crps


def center_cross_keep(mask: np.ndarray, cross_index: int) -> np.ndarray:
    """Return a mask in packed-composition order, excluding one grid row/column."""
    if mask.ndim != 2 or not (0 <= cross_index < min(mask.shape)):
        raise ValueError(f"invalid mask shape/cross index: {mask.shape}, {cross_index}")
    cross = np.zeros(mask.shape, dtype=bool)
    cross[cross_index, :] = True
    cross[:, cross_index] = True
    return (~cross)[~mask]


def normalized_best(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    distances: dict[str, np.ndarray] = {}
    for column, higher_is_better in (("tau", True), ("mse", False), ("crps", False)):
        values = np.asarray([float(row[column]) for row in rows])
        low, high = float(np.min(values)), float(np.max(values))
        if high - low < 1.0e-14:
            distances[column] = np.zeros_like(values)
        elif higher_is_better:
            distances[column] = (high - values) / (high - low)
        else:
            distances[column] = (values - low) / (high - low)
    scores = np.sqrt(sum(value * value for value in distances.values()) / 3.0)
    # Same ordering as the legacy selector: score, high tau, low MSE, low CRPS.
    index = min(
        range(len(rows)),
        key=lambda i: (scores[i], -float(rows[i]["tau"]), float(rows[i]["mse"]), float(rows[i]["crps"])),
    )
    selected = dict(rows[index])
    selected.update(
        d_tau=float(distances["tau"][index]),
        d_mse=float(distances["mse"][index]),
        d_crps=float(distances["crps"][index]),
        best_score=float(scores[index]),
    )
    return selected


def trial_number(path: Path) -> int:
    return int(path.name.removeprefix("trial_"))


def expected_trials(shard: Path) -> set[int]:
    return {int(path.stem.removeprefix("trial_")) for path in (shard / "packed_atoms").glob("trial_*.3bit")}


def completed_trials(shard: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for marker in (shard / "trial_zarr").glob("trial_*/.activity_complete"):
        result[trial_number(marker.parent)] = marker.parent
    return result


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--experimental", type=Path, default=Path("orr_log_matched_activity.json"))
    parser.add_argument("--mask", type=Path, default=Path(
        "/home/ktg0829/project/HEA/CEMC/CEMC_new_composition/static_grid_mask.npy"
    ))
    parser.add_argument("--cross-index", type=int, default=25)
    parser.add_argument("--include-center-cross", action="store_true", help="Use all 1400 compositions, as in the bundled histogram score tables")
    parser.add_argument("--output-dir", type=Path, default=Path("results/crps_selection_activity_log"))
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--trial-min", type=int)
    parser.add_argument("--trial-max", type=int)
    parser.add_argument("--allow-incomplete", action="store_true", help="score completed trials even if jobs are running")
    args = parser.parse_args()

    with args.experimental.open(encoding="utf-8") as handle:
        records = json.load(handle)
    keys = sorted(int(key) for key in records)
    if keys != list(range(len(keys))):
        raise SystemExit("experimental JSON keys must be contiguous composition indices starting at 0")
    experimental = np.asarray([records[str(i)] for i in keys], dtype=np.float64)
    if not np.isfinite(experimental).all():
        raise SystemExit("experimental activity contains non-finite values")
    static_mask = np.load(args.mask).astype(bool)
    evaluation_keep = (np.ones(int(np.count_nonzero(~static_mask)), dtype=bool) if args.include_center_cross else center_cross_keep(static_mask, args.cross_index))
    if len(evaluation_keep) != len(experimental):
        raise SystemExit(
            f"mask/experimental mismatch: mask has {len(evaluation_keep)} packed pixels, "
            f"JSON has {len(experimental)}"
        )
    experimental = experimental[evaluation_keep]

    trial_stores: list[tuple[int, Path, str]] = []
    missing_total = 0
    for shard_index in range(args.world_size):
        shard = args.results_root / f"shard_{shard_index:02d}"
        expected = expected_trials(shard)
        complete = completed_trials(shard)
        missing = expected - complete.keys()
        missing_total += len(missing)
        print(f"{shard.name}: {len(complete)}/{len(expected)} activity trials complete", flush=True)
        for trial, store in complete.items():
            if args.trial_min is not None and trial < args.trial_min:
                continue
            if args.trial_max is not None and trial > args.trial_max:
                continue
            trial_stores.append((trial, store, shard.name))
    if missing_total and not args.allow_incomplete:
        raise SystemExit(
            f"activity calculation is incomplete: {missing_total} trial(s) are missing; "
            "wait for the array job or pass --allow-incomplete intentionally"
        )
    if not trial_stores:
        raise SystemExit("no completed activity trials found")
    trial_stores.sort()

    cemc_rows: list[dict[str, float | int]] = []
    best_rows: list[dict[str, float | int]] = []
    random_rows: list[dict[str, float | int]] = []
    for number, (trial, root, source_shard) in enumerate(trial_stores, start=1):
        with (root / "mc.zarr" / ".zattrs").open(encoding="utf-8") as handle:
            temperatures = [int(round(float(x))) for x in json.load(handle)["temperatures"]]
        mc_mean = read_zarr_array(root / "mc.zarr", "activity_mean")
        mc_sd = read_zarr_array(root / "mc.zarr", "activity_sd")
        if mc_mean.shape[0] != len(temperatures):
            raise ValueError(f"temperature shape mismatch in {root / 'mc.zarr'}")
        one_trial: list[dict[str, float | int]] = []
        for index, temperature in enumerate(temperatures):
            tau, mse, crps = metrics(
                mc_mean[index][evaluation_keep], mc_sd[index][evaluation_keep], experimental
            )
            row: dict[str, float | int] = {
                "trial": trial,
                "temperature": temperature,
                "crps": crps,
                "tau": tau,
                "mse": mse,
                "n_compositions": len(experimental),
                "source_shard": source_shard,
            }
            one_trial.append(row)
            cemc_rows.append(row)
        best_rows.append(normalized_best(one_trial))

        random_mean = read_zarr_array(root / "random.zarr", "activity_mean")
        random_sd = read_zarr_array(root / "random.zarr", "activity_sd")
        tau, mse, crps = metrics(
            random_mean[evaluation_keep], random_sd[evaluation_keep], experimental
        )
        random_rows.append({
            "trial": trial,
            "crps": crps,
            "tau": tau,
            "mse": mse,
            "n_compositions": len(experimental),
            "source_shard": source_shard,
        })
        if number % 100 == 0 or number == len(trial_stores):
            print(f"scored {number}/{len(trial_stores)} trials", flush=True)

    common = ["trial", "temperature", "crps", "tau", "mse", "n_compositions", "source_shard"]
    write_csv_atomic(args.output_dir / "crps_by_trial_temperature.csv", common, cemc_rows)
    write_csv_atomic(
        args.output_dir / "best_temperature_by_trial_crps.csv",
        common + ["d_tau", "d_mse", "d_crps", "best_score"],
        best_rows,
    )
    write_csv_atomic(
        args.output_dir / "random_crps_by_trial.csv",
        ["trial", "crps", "tau", "mse", "n_compositions", "source_shard"],
        random_rows,
    )
    metadata = {
        "activity_domain": "log",
        "experimental": str(args.experimental.resolve()),
        "n_trials": len(trial_stores),
        "n_compositions": len(experimental),
        "excluded_center_cross_index": None if args.include_center_cross else args.cross_index,
        "n_cross_pixels_excluded": int(np.count_nonzero(~evaluation_keep)),
        "metric_scaling": "prediction independently min-max scaled to [-1, 0]",
        "crps": "Gaussian CRPS using run SD propagated through min-max scaling",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote log-domain scores to {args.output_dir}")


if __name__ == "__main__":
    main()
