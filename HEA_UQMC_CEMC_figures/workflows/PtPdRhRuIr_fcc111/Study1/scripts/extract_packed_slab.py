#!/usr/bin/env python3
"""Extract one 3-bit packed slab and optionally write it with ASE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unpack_codes(data: bytes, n_atoms: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    codes = np.empty(n_atoms, dtype=np.uint8)
    for atom in range(n_atoms):
        bit = atom * 3
        byte, shift = divmod(bit, 8)
        value = int(raw[byte]) >> shift
        if shift > 5:
            value |= int(raw[byte + 1]) << (8 - shift)
        codes[atom] = value & 0b111
    if np.any(codes > 4):
        raise ValueError("Packed slab contains an invalid element code (valid range: 0..4)")
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--composition", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--method", choices=["random", "cemc"], required=True)
    parser.add_argument("--temp-index", type=int, default=None)
    parser.add_argument("--template", default="template_fcc111_10x10x10.cif")
    parser.add_argument("--output", required=True, help="Output supported by ASE, e.g. slab.cif")
    args = parser.parse_args()

    packed_dir = Path(args.results_root) / "packed_atoms"
    metadata = json.loads((packed_dir / "metadata.json").read_text())
    ncomp = int(metadata["n_compositions"])
    nruns = int(metadata["n_runs"])
    ntemps = int(metadata["n_temperatures"])
    n_atoms = int(metadata["n_atoms"])
    slab_bytes = int(metadata["bytes_per_slab"])

    if not 0 <= args.composition < ncomp:
        raise SystemExit(f"--composition must be in [0, {ncomp - 1}]")
    if not 0 <= args.run < nruns:
        raise SystemExit(f"--run must be in [0, {nruns - 1}]")
    if args.method == "random":
        if args.temp_index is not None:
            raise SystemExit("Random slabs do not have a temperature index")
        state = 0
    else:
        if args.temp_index is None or not 0 <= args.temp_index < ntemps:
            raise SystemExit(f"CEMC requires --temp-index in [0, {ntemps - 1}]")
        state = args.temp_index + 1

    task = args.composition * nruns + args.run
    offset = (task * (ntemps + 1) + state) * slab_bytes
    path = packed_dir / f"trial_{args.trial:06d}.3bit"
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(slab_bytes)
    if len(data) != slab_bytes:
        raise SystemExit(f"Packed slab is missing or truncated in {path}")

    codes = unpack_codes(data, n_atoms)
    code_to_z = np.asarray(metadata["code_to_atomic_number"], dtype=np.int64)
    atomic_numbers = code_to_z[codes]

    from ase.io import read, write

    atoms = read(args.template)
    if len(atoms) != n_atoms:
        raise SystemExit(f"Template has {len(atoms)} atoms; packed slab has {n_atoms}")
    atoms.numbers[:] = atomic_numbers
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
