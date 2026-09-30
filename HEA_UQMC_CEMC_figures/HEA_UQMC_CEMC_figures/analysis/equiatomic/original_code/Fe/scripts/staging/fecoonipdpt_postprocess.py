#!/usr/bin/env python3
"""Create and validate CEMC, initial, and within-layer-shuffled CIF slabs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write


BASE = Path("/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic")
RECONSTRUCT_EXE = BASE / "bin" / "reconstruct_random_slab"
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")
EXPECTED_COUNTS = {element: 200 for element in ELEMENTS}
TEMPERATURES = tuple(range(2000, 299, -100)) + (298,)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def layer_groups(atoms) -> list[np.ndarray]:
    keys = np.round(atoms.positions[:, 2], decimals=6)
    groups = [np.flatnonzero(keys == value) for value in sorted(set(keys))]
    if len(groups) != 10 or any(len(group) != 100 for group in groups):
        raise RuntimeError(f"Expected 10 layers x 100 atoms; found {[len(g) for g in groups]}")
    return groups


def verify_counts(atoms, context: str) -> None:
    symbols = atoms.get_chemical_symbols()
    counts = {element: symbols.count(element) for element in ELEMENTS}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"{context}: wrong composition {counts}")


def atoms_from_record(template, record: dict):
    atoms = template.copy()
    numbers = np.asarray(record["Z"], dtype=int)
    if len(numbers) != len(atoms):
        raise RuntimeError(f"Expected {len(atoms)} atomic numbers, found {len(numbers)}")
    atoms.set_atomic_numbers(numbers)
    atoms.info.update({
        "run": int(record.get("run", -1)),
        "target_temperature_K": float(record.get("target_temperature", -1)),
        "actual_temperature_K": float(record.get("actual_temperature", -1)),
        "mc_step": int(record.get("mc_step", -1)),
        "energy_eV": float(record.get("energy", 0.0)),
    })
    return atoms


def shuffled_within_layers(atoms, groups: list[np.ndarray], seed: int):
    shuffled = atoms.copy()
    numbers = shuffled.get_atomic_numbers()
    rng = np.random.default_rng(seed)
    for group in groups:
        values = numbers[group].copy()
        rng.shuffle(values)
        numbers[group] = values
    shuffled.set_atomic_numbers(numbers)
    shuffled.info["layer_shuffle_seed"] = int(seed)
    return shuffled


def load_records(workdir: Path) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    for temperature in TEMPERATURES:
        path = workdir / "results" / "structures_by_temperature" / f"T{temperature:05d}" / "comp_0000.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (int(record["run"]), temperature)
            if key in records:
                raise RuntimeError(f"Duplicate snapshot {key}")
            records[key] = record
    expected = 20 * len(TEMPERATURES)
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} snapshots, found {len(records)}")
    return records


def reconstruct_initial(workdir: Path, run: int, raw: Path) -> dict:
    command = [
        str(RECONSTRUCT_EXE),
        "--ce-export", str(workdir / "inputs" / "ce_export.txt"),
        "--seeds", str(workdir / "results" / "random_slab_seeds.csv"),
        "--trial", "0", "--composition-index", "0", "--run", str(run),
        "--output", str(raw),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    numbers = json.loads(raw.read_text())
    raw.unlink()
    return {
        "run": run, "target_temperature": -1, "actual_temperature": -1,
        "mc_step": 0, "energy": 0.0, "Z": numbers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facet", required=True, choices=("100", "110", "111"))
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    combo = f"{args.facet}/{args.site}"
    workdir = BASE / args.facet / args.site
    template = read(workdir / "inputs" / "template.cif")
    template.pbc = True
    groups = layer_groups(template)
    records = load_records(workdir)
    slabs_dir = workdir / "slabs"
    rows: list[dict] = []

    warnings.filterwarnings("ignore", message="Occupancies present but no occupancy info")
    for run in range(20):
        slab_id = run + 1
        outdir = slabs_dir / f"slab_{slab_id:02d}"
        outdir.mkdir(parents=True, exist_ok=True)
        initial_record = reconstruct_initial(workdir, run, outdir / "initial_random.json")
        initial = atoms_from_record(template, initial_record)
        verify_counts(initial, f"{combo} slab {slab_id} initial")
        write(outdir / "initial_random.cif", initial)

        for temperature in TEMPERATURES:
            record = records[(run, temperature)]
            atoms = atoms_from_record(template, record)
            verify_counts(atoms, f"{combo} slab {slab_id} T={temperature}")
            final = temperature == 298
            stem = "final_0298K" if final else f"T_{temperature:04d}K"
            cemc_path = outdir / f"{stem}_cemc.cif"
            shuffled_path = outdir / f"{stem}_layer_shuffled.cif"
            write(cemc_path, atoms)

            seed = stable_seed("FeCoNiPdPt-equi-layer-shuffle-v1", combo, run, temperature)
            shuffled = shuffled_within_layers(atoms, groups, seed)
            verify_counts(shuffled, f"{combo} slab {slab_id} T={temperature} shuffled")
            before = atoms.get_atomic_numbers()
            after = shuffled.get_atomic_numbers()
            for layer_index, group in enumerate(groups):
                if sorted(before[group].tolist()) != sorted(after[group].tolist()):
                    raise RuntimeError(f"Cross-layer mixing: {combo} slab {slab_id}, layer {layer_index}")
            write(shuffled_path, shuffled)
            rows.append({
                "facet": args.facet, "site": args.site, "slab": slab_id, "run_index": run,
                "target_temperature_K": temperature,
                "actual_temperature_K": record["actual_temperature"], "mc_step": record["mc_step"],
                "energy_eV": record["energy"], "attempted": record["attempted"],
                "accepted": record["accepted"], "layer_shuffle_seed": seed,
                "cemc_file": str(cemc_path.relative_to(workdir)),
                "layer_shuffled_file": str(shuffled_path.relative_to(workdir)),
            })

    with (workdir / "structure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "facet": args.facet, "site": args.site, "n_independent_slabs": 20,
        "n_atoms_per_slab": 1000, "composition_counts": EXPECTED_COUNTS,
        "layers": 10, "atoms_per_layer": 100,
        "snapshot_temperatures_K": list(range(2000, 299, -100)),
        "final_temperature_K": 298,
        "mc_schedule": "2023 K hold for 10 sweeps, then exponential 2023 K to 298 K for 100 sweeps",
        "files_per_slab": 39, "total_cif_files": 780,
    }
    (workdir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
