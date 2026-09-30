#!/usr/bin/env python3
"""Calculate OH activities directly from packed *.3bit UQMC trajectories.

The output intentionally matches the activity portion of the legacy UQMC Zarr
layout consumed by plot_tau_mse_crps_modal_activity_map_kde.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


KB_EV_PER_K = 8.617333262145e-5
KB_SI = 1.38e-23
H_SI = 6.626e-34


@dataclass(frozen=True)
class Model:
    elements: tuple[str, ...]
    intercept: float
    zone1: np.ndarray
    zone2: np.ndarray
    zone3: np.ndarray
    e_opt: float
    temperature: float
    surface_sites: np.ndarray
    zone2_ptr: np.ndarray
    zone2_indices: np.ndarray
    zone3_ptr: np.ndarray
    zone3_indices: np.ndarray


def parse_model(path: Path, topology_path: Path) -> Model:
    values: dict[str, list[str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ACTIVITY_MODEL_OH_V1":
        raise ValueError(f"unsupported activity model: {path}")
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        key, data = parts[0], parts[1:]
        if key in {"zone2_indices", "zone3_indices"}:
            count = int(data[0])
            data = []
            while len(data) < count and i < len(lines):
                data.extend(lines[i].split())
                i += 1
            if len(data) != count:
                raise ValueError(f"{key}: expected {count} entries, got {len(data)}")
        values[key] = data
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    surface = np.asarray(topology["surface_sites"], dtype=np.int64)
    model = Model(
        elements=tuple(values["elements"]),
        intercept=float(values["intercept"][0]),
        zone1=np.asarray(values["zone1"], dtype=np.float64),
        zone2=np.asarray(values["zone2"], dtype=np.float64),
        zone3=np.asarray(values["zone3"], dtype=np.float64),
        e_opt=float(values["e_opt"][0]),
        temperature=float(values["activity_temperature"][0]),
        surface_sites=surface,
        zone2_ptr=np.asarray(values["zone2_ptr"], dtype=np.int64),
        zone2_indices=np.asarray(values["zone2_indices"], dtype=np.int64),
        zone3_ptr=np.asarray(values["zone3_ptr"], dtype=np.int64),
        zone3_indices=np.asarray(values["zone3_indices"], dtype=np.int64),
    )
    if len(model.zone2_ptr) != len(surface) + 1 or len(model.zone3_ptr) != len(surface) + 1:
        raise ValueError("activity topology pointer size mismatch")
    if set(model.elements) != {"Ru", "Rh", "Pd", "Ir", "Pt"}:
        raise ValueError(f"unexpected activity elements: {model.elements}")
    return model


def read_temperatures(path: Path, expected: int) -> list[float]:
    tokens = path.read_text(encoding="utf-8").split()
    try:
        pos = tokens.index("target_temperatures")
    except ValueError as exc:
        raise ValueError(f"target_temperatures missing from {path}") from exc
    temperatures = [float(x) for x in tokens[pos + 1 : pos + 1 + expected]]
    if len(temperatures) != expected:
        raise ValueError(f"expected {expected} temperatures in {path}")
    return temperatures


def read_shifts(path: Path) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result[int(row["trial"])] = np.asarray(
                [row[f"be_shift_{e}"] for e in ("Ir", "Pd", "Pt", "Rh", "Ru")],
                dtype=np.float64,
            )
    return result


def decode_atoms(packed: np.ndarray, atom_indices: np.ndarray) -> np.ndarray:
    """Decode selected atoms from rows of little-endian 3-bit slabs."""
    bit = atom_indices * 3
    byte = bit >> 3
    offset = bit & 7
    lo = packed[:, byte].astype(np.uint16)
    hi_byte = np.minimum(byte + 1, packed.shape[1] - 1)
    hi = packed[:, hi_byte].astype(np.uint16)
    words = lo | (hi << 8)
    codes = ((words >> offset) & 7).astype(np.int8)
    if np.any(codes > 4):
        raise ValueError("packed slab contains element code > 4")
    return codes


def evaluate_block(
    packed: np.ndarray,
    model: Model,
    shifts: np.ndarray,
    packed_to_model: np.ndarray,
) -> np.ndarray:
    needed = np.unique(
        np.concatenate((model.surface_sites, model.zone2_indices, model.zone3_indices))
    )
    decoded = decode_atoms(packed, needed)
    lookup = np.full(int(needed[-1]) + 1, -1, dtype=np.int64)
    lookup[needed] = np.arange(len(needed))
    occupations = packed_to_model[decoded]
    top = occupations[:, lookup[model.surface_sites]]
    energy = model.intercept + model.zone1[top]
    for sidx in range(len(model.surface_sites)):
        z2 = model.zone2_indices[model.zone2_ptr[sidx] : model.zone2_ptr[sidx + 1]]
        z3 = model.zone3_indices[model.zone3_ptr[sidx] : model.zone3_ptr[sidx + 1]]
        if len(z2):
            energy[:, sidx] += model.zone2[occupations[:, lookup[z2]]].sum(axis=1)
        if len(z3):
            energy[:, sidx] += model.zone3[occupations[:, lookup[z3]]].sum(axis=1)
    energy -= shifts[top]
    k_t = KB_EV_PER_K * model.temperature
    log_site = math.log(KB_SI * model.temperature / H_SI) - np.abs(energy - model.e_opt) / k_t
    maximum = np.max(log_site, axis=1)
    return maximum + np.log(np.exp(log_site - maximum[:, None]).sum(axis=1)) - math.log(log_site.shape[1])


def write_zarr_array(group: Path, name: str, data: np.ndarray) -> None:
    array_dir = group / name
    array_dir.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(data, dtype="<f4")
    metadata = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(data.shape),
        "dtype": "<f4",
        "compressor": None,
        "fill_value": None,
        "order": "C",
        "filters": None,
    }
    (array_dir / ".zarray").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    chunk_name = ".".join("0" for _ in data.shape)
    data.tofile(array_dir / chunk_name)


def write_group(group: Path, trial: int, temperatures: list[float] | None, arrays: dict[str, np.ndarray]) -> None:
    group.mkdir(parents=True, exist_ok=True)
    (group / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    attrs: dict[str, object] = {"trial": trial, "activity_postprocessed_from_3bit": True}
    if temperatures is not None:
        attrs["temperatures"] = temperatures
    (group / ".zattrs").write_text(json.dumps(attrs, indent=2) + "\n", encoding="utf-8")
    for name, data in arrays.items():
        write_zarr_array(group, name, data)


def sample_sd(data: np.ndarray, axis: int) -> np.ndarray:
    if data.shape[axis] < 2:
        return np.zeros(np.delete(data.shape, axis), dtype=np.float64)
    return np.std(data, axis=axis, ddof=1)


def process_trial(
    packed_path_text: str,
    output_shard_text: str,
    model_path_text: str,
    topology_path_text: str,
    metadata: dict[str, object],
    temperatures: list[float],
    shifts: np.ndarray,
    overwrite: bool,
    max_compositions: int | None,
) -> tuple[int, str, np.ndarray, np.ndarray]:
    packed_path = Path(packed_path_text)
    output_shard = Path(output_shard_text)
    match = re.fullmatch(r"trial_(\d+)\.3bit", packed_path.name)
    if not match:
        raise ValueError(f"invalid trial filename: {packed_path}")
    trial = int(match.group(1))
    final_root = output_shard / "trial_zarr" / f"trial_{trial:06d}"
    done = final_root / ".activity_complete"
    if done.exists() and not overwrite:
        summary = np.load(final_root / "activity_summary.npz")
        return trial, "skipped", summary["random_mean"], summary["random_sd"]

    model = parse_model(Path(model_path_text), Path(topology_path_text))
    ncomp = int(metadata["n_compositions"])
    nruns = int(metadata["n_runs"])
    ntemps = int(metadata["n_temperatures"])
    nstates = ntemps + 1
    bytes_per_slab = int(metadata["bytes_per_slab"])
    expected_size = ncomp * nruns * nstates * bytes_per_slab
    actual_size = packed_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{packed_path}: expected {expected_size} bytes, got {actual_size}")
    work_ncomp = ncomp if max_compositions is None else min(ncomp, max_compositions)
    raw = np.memmap(packed_path, mode="r", dtype=np.uint8).reshape(ncomp, nruns * nstates, bytes_per_slab)
    cemc_mean = np.empty((ntemps, work_ncomp), dtype=np.float32)
    cemc_sd = np.empty_like(cemc_mean)
    random_mean = np.empty(work_ncomp, dtype=np.float32)
    random_sd = np.empty_like(random_mean)
    packed_symbols = tuple(metadata["code_to_symbol"])
    packed_to_model = np.asarray([model.elements.index(e) for e in packed_symbols], dtype=np.int8)
    for comp in range(work_ncomp):
        activities = evaluate_block(raw[comp], model, shifts, packed_to_model).reshape(nruns, nstates)
        random_mean[comp] = np.mean(activities[:, 0])
        random_sd[comp] = sample_sd(activities[:, 0], axis=0)
        cemc_mean[:, comp] = np.mean(activities[:, 1:], axis=0)
        cemc_sd[:, comp] = sample_sd(activities[:, 1:], axis=0)

    temp_parent = output_shard / "trial_zarr"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".trial_{trial:06d}_", dir=temp_parent))
    try:
        write_group(temp_root / "mc.zarr", trial, temperatures, {"activity_mean": cemc_mean, "activity_sd": cemc_sd})
        write_group(temp_root / "random.zarr", trial, None, {"activity_mean": random_mean, "activity_sd": random_sd})
        np.savez(temp_root / "activity_summary.npz", random_mean=random_mean, random_sd=random_sd)
        (temp_root / ".activity_complete").write_text("activity_v1\n", encoding="utf-8")
        if final_root.exists():
            if not overwrite:
                raise FileExistsError(final_root)
            shutil.rmtree(final_root)
        os.replace(temp_root, final_root)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    return trial, "written", random_mean, random_sd


def rewrite_random_summary(output_shard: Path) -> None:
    rows: list[tuple[int, int, float, float]] = []
    for summary_path in sorted((output_shard / "trial_zarr").glob("trial_*/activity_summary.npz")):
        trial = int(summary_path.parent.name.removeprefix("trial_"))
        summary = np.load(summary_path)
        for comp, (mean, sd) in enumerate(zip(summary["random_mean"], summary["random_sd"])):
            rows.append((trial, comp, float(mean), float(sd)))
    target = output_shard / "predicted_activity_selected_by_trial.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trial", "composition_index", "random_pred_activity_mean", "random_pred_activity_sd"))
        writer.writerows(rows)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--activity-model", type=Path, default=Path("activity_model_oh.txt"))
    parser.add_argument("--topology", type=Path, default=Path("activity_topology.json"))
    parser.add_argument("--schedule", type=Path, default=Path("schedule_snapshots.txt"))
    parser.add_argument("--be-shifts", type=Path, default=Path("results/shift_metadata/trial_be_shifts.csv"))
    parser.add_argument("--shard", action="append", help="shard name (repeatable; default: all shard_*)")
    parser.add_argument("--trial", type=int, action="append", help="trial number (repeatable; default: all)")
    parser.add_argument("--workers", type=int, default=1, help="parallel trial workers per shard")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-compositions", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    shifts_by_trial = read_shifts(args.be_shifts)
    shards = [args.results_root / name for name in args.shard] if args.shard else sorted(args.results_root.glob("shard_*"))
    if not shards:
        raise SystemExit(f"no shards found under {args.results_root}")
    requested = set(args.trial or [])
    failures = 0
    for shard in shards:
        metadata_path = shard / "packed_atoms" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        temperatures = read_temperatures(args.schedule, int(metadata["n_temperatures"]))
        packed_files = sorted((shard / "packed_atoms").glob("trial_*.3bit"))
        if requested:
            packed_files = [p for p in packed_files if int(p.stem.split("_")[1]) in requested]
        jobs = []
        for packed_path in packed_files:
            trial = int(packed_path.stem.split("_")[1])
            if trial not in shifts_by_trial:
                print(f"ERROR trial {trial}: BE shifts missing", file=sys.stderr)
                failures += 1
                continue
            jobs.append((str(packed_path), str(shard), str(args.activity_model), str(args.topology), metadata,
                         temperatures, shifts_by_trial[trial], args.overwrite, args.max_compositions))
        print(f"{shard.name}: {len(jobs)} trial(s), workers={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_trial, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                try:
                    trial, status, _, _ = future.result()
                    print(f"{shard.name} trial {trial}: {status}", flush=True)
                except Exception as exc:
                    failures += 1
                    print(f"ERROR {futures[future]}: {exc}", file=sys.stderr, flush=True)
        rewrite_random_summary(shard)
    if failures:
        raise SystemExit(f"{failures} trial(s) failed")


if __name__ == "__main__":
    main()
