#!/usr/bin/env python3
from __future__ import annotations

import configparser
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read


TEMPERATURES = tuple(range(2000, 299, -10)) + (298,)
N_RUNS = 20
N_ATOMS = 1000


@dataclass(frozen=True)
class Case:
    name: str
    original_config: Path
    binary: Path
    template: Path
    output_root: Path
    expected_atomic_numbers: tuple[int, ...]


def write_schedule(path: Path) -> None:
    path.write_text(
        "SCHEDULE_SNAPSHOTS_V1\n"
        f"n_targets {len(TEMPERATURES)}\n"
        "target_temperatures " + " ".join(map(str, TEMPERATURES)) + "\n"
        "n_segments 2\n"
        "segment 0 2023 2023 10000\n"
        "segment 2 2023 298 100000\n"
    )


def rewrite_config(original: Path, destination: Path, schedule: Path, results: Path) -> None:
    lines = []
    found_schedule = found_output = False
    for raw in original.read_text().splitlines():
        key = raw.split("=", 1)[0].strip() if "=" in raw else ""
        if key == "schedule_export":
            lines.append(f"schedule_export = {schedule}")
            found_schedule = True
        elif key == "output_dir":
            lines.append(f"output_dir = {results}")
            found_output = True
        else:
            lines.append(raw)
    if not (found_schedule and found_output):
        raise RuntimeError(f"Could not rewrite schedule/output in {original}")
    destination.write_text("\n".join(lines) + "\n")


def load_records(results: Path):
    z = np.empty((len(TEMPERATURES), N_RUNS, N_ATOMS), dtype=np.uint8)
    energy = np.empty((len(TEMPERATURES), N_RUNS), dtype=np.float64)
    actual_temperature = np.empty((len(TEMPERATURES), N_RUNS), dtype=np.float64)
    mc_step = np.empty((len(TEMPERATURES), N_RUNS), dtype=np.int64)
    attempted = np.empty((len(TEMPERATURES), N_RUNS), dtype=np.int64)
    accepted = np.empty((len(TEMPERATURES), N_RUNS), dtype=np.int64)
    for tidx, temperature in enumerate(TEMPERATURES):
        path = results / "structures_by_temperature" / f"T{temperature:05d}" / "comp_0000.jsonl"
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["run"])] = row
        if sorted(rows) != list(range(N_RUNS)):
            raise RuntimeError(f"{path}: missing runs {sorted(rows)}")
        for run in range(N_RUNS):
            row = rows[run]
            values = np.asarray(row["Z"], dtype=np.uint8)
            if values.shape != (N_ATOMS,):
                raise RuntimeError(f"{path} run {run}: Z shape {values.shape}")
            z[tidx, run] = values
            energy[tidx, run] = float(row["energy"])
            actual_temperature[tidx, run] = float(row["actual_temperature"])
            mc_step[tidx, run] = int(row["mc_step"])
            attempted[tidx, run] = int(row["attempted"])
            accepted[tidx, run] = int(row["accepted"])
    return z, energy, actual_temperature, mc_step, attempted, accepted


def validate_composition(z: np.ndarray, expected: tuple[int, ...], case_name: str) -> None:
    wanted = np.asarray(expected, dtype=np.uint8)
    for atomic_number in wanted:
        counts = np.sum(z == atomic_number, axis=2)
        if not np.all(counts == 200):
            raise RuntimeError(
                f"{case_name}: Z={int(atomic_number)} composition range "
                f"{int(counts.min())}..{int(counts.max())}, expected 200"
            )
    observed = np.unique(z)
    if not np.array_equal(np.sort(observed), np.sort(wanted)):
        raise RuntimeError(f"{case_name}: unexpected atomic numbers {observed.tolist()}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(case: Case) -> None:
    case.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stage = case.output_root / f"run_work_{stamp}"
    stage.mkdir()
    schedule = stage / "schedule_10K.txt"
    config = stage / "run_config_10K.ini"
    results = stage / "results"
    write_schedule(schedule)
    rewrite_config(case.original_config, config, schedule, results)

    subprocess.run(
        ["mpirun", "-np", "12", str(case.binary), "--config", str(config)],
        check=True,
    )
    z, energy, actual_temperature, mc_step, attempted, accepted = load_records(results)
    validate_composition(z, case.expected_atomic_numbers, case.name)

    template = read(case.template)
    if len(template) != N_ATOMS:
        raise RuntimeError(f"{case.template}: expected {N_ATOMS} atoms, found {len(template)}")
    archive = case.output_root / "cemc_structures_10K.npz"
    archive_tmp = case.output_root / f".{archive.name}.{stamp}.tmp"
    with archive_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            Z=z,
            temperatures_K=np.asarray(TEMPERATURES, dtype=np.int16),
            runs=np.arange(N_RUNS, dtype=np.int16),
            energy_eV=energy,
            actual_temperature_K=actual_temperature,
            mc_step=mc_step,
            attempted=attempted,
            accepted=accepted,
            cell_A=np.asarray(template.cell.array, dtype=np.float64),
            scaled_positions=np.asarray(template.get_scaled_positions(wrap=False), dtype=np.float32),
            pbc=np.asarray(template.pbc, dtype=np.bool_),
        )

    with np.load(archive_tmp) as packed:
        if packed["Z"].shape != (len(TEMPERATURES), N_RUNS, N_ATOMS):
            raise RuntimeError(f"{case.name}: packed Z shape {packed['Z'].shape}")
        if not np.array_equal(packed["temperatures_K"], np.asarray(TEMPERATURES)):
            raise RuntimeError(f"{case.name}: packed temperature mismatch")
        validate_composition(packed["Z"], case.expected_atomic_numbers, case.name + " packed")
    os.replace(archive_tmp, archive)

    final_schedule = case.output_root / "schedule_10K.txt"
    final_config = case.output_root / "run_config_10K.ini"
    shutil.copy2(schedule, final_schedule)
    # Store the exact run config for provenance even though its raw output path
    # is intentionally removed after packing.
    shutil.copy2(config, final_config)
    digest = sha256(archive)
    metadata = {
        "case": case.name,
        "format": "NumPy compressed NPZ",
        "lossless": True,
        "archive": archive.name,
        "sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "array_shape_Z": [len(TEMPERATURES), N_RUNS, N_ATOMS],
        "array_dtype_Z": "uint8 atomic numbers",
        "temperature_count": len(TEMPERATURES),
        "temperatures_K": list(TEMPERATURES),
        "temperature_spacing": "10 K from 2000 K through 300 K, plus 298 K",
        "n_runs": N_RUNS,
        "n_atoms": N_ATOMS,
        "composition_per_structure": {str(zv): 200 for zv in case.expected_atomic_numbers},
        "original_config": str(case.original_config),
        "template_source": str(case.template),
        "reconstruction": (
            "with np.load('cemc_structures_10K.npz') as d: "
            "atoms = ase.Atoms(numbers=d['Z'][tidx,run], "
            "scaled_positions=d['scaled_positions'], cell=d['cell_A'], pbc=d['pbc'])"
        ),
    }
    metadata_tmp = case.output_root / f".metadata.{stamp}.tmp"
    metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(metadata_tmp, case.output_root / "metadata.json")
    shutil.rmtree(stage)
    print(
        f"PACKED {case.name}: {archive} "
        f"({archive.stat().st_size / 1024 / 1024:.2f} MiB, sha256={digest})",
        flush=True,
    )


def main() -> None:
    pt = Path("/home/jinsookim/HEA_MC/PtPdRhRuIr/equi-atomic")
    fe = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
    cases = (
        Case(
            "PtPdIrRhRu fcc(111)",
            pt / "inputs/equi_atomic.ini",
            pt / "bin/uq_cemc_mpi",
            pt / "inputs/template_fcc111_10x10x10.cif",
            pt / "compact_cemc_10K/fcc111",
            (78, 46, 77, 45, 44),
        ),
        Case(
            "FeCoNiPdPt fcc(100)",
            fe / "100/bridge/inputs/equi_atomic.ini",
            fe / "100/bridge/bin/uq_cemc_mpi",
            fe / "100/bridge/inputs/template.cif",
            fe / "compact_cemc_10K/fcc100",
            (26, 27, 28, 46, 78),
        ),
        Case(
            "FeCoNiPdPt fcc(110)",
            fe / "110/bridge/inputs/equi_atomic.ini",
            fe / "110/bridge/bin/uq_cemc_mpi",
            fe / "110/bridge/inputs/template.cif",
            fe / "compact_cemc_10K/fcc110",
            (26, 27, 28, 46, 78),
        ),
        Case(
            "FeCoNiPdPt fcc(111)",
            fe / "111/hollow/inputs/equi_atomic.ini",
            fe / "111/hollow/bin/uq_cemc_mpi",
            fe / "111/hollow/inputs/template.cif",
            fe / "compact_cemc_10K/fcc111",
            (26, 27, 28, 46, 78),
        ),
    )
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
