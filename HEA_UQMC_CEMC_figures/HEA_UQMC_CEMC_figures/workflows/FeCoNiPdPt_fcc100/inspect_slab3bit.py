#!/usr/bin/env python3
"""Inspect and validate fixed-length 3-bit CEMC slab shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

MAGIC = b"C3SLAB01"


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = handle.read(16)
        if len(prefix) != 16 or prefix[:8] != MAGIC:
            raise ValueError(f"{path}: invalid C3SLAB01 header")
        version, json_size = struct.unpack("<II", prefix[8:])
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        payload = handle.read(json_size)
    header = json.loads(payload)
    expected = (
        int(header["header_bytes"])
        + int(header["n_trials_in_shard"])
        * int(header["n_compositions"])
        * int(header["n_runs"])
        * int(header["n_temperatures"])
        * int(header["bytes_per_slab"])
    )
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: size {actual}, expected {expected}")
    return header


def unpack_3bit(data: bytes, n_sites: int) -> list[int]:
    values: list[int] = []
    for site in range(n_sites):
        bit = 3 * site
        byte = bit >> 3
        shift = bit & 7
        value = (data[byte] >> shift) & 0b111
        if shift > 5:
            value |= (data[byte + 1] << (8 - shift)) & 0b111
        if value >= 5:
            raise ValueError(f"invalid packed element code {value} at site {site}")
        values.append(value)
    return values


def read_slab(
    path: Path,
    header: dict,
    trial: int,
    composition: int,
    run: int,
    temperature_index: int,
) -> list[int]:
    local_trial = trial - int(header["first_trial"])
    dims = (
        int(header["n_trials_in_shard"]),
        int(header["n_compositions"]),
        int(header["n_runs"]),
        int(header["n_temperatures"]),
    )
    indices = (local_trial, composition, run, temperature_index)
    for name, index, size in zip(
        ("trial", "composition", "run", "temperature_index"), indices, dims
    ):
        if not 0 <= index < size:
            raise IndexError(f"{name}={index} outside [0,{size})")
    record = (
        ((local_trial * dims[1] + composition) * dims[2] + run) * dims[3]
        + temperature_index
    )
    nbytes = int(header["bytes_per_slab"])
    offset = int(header["header_bytes"]) + record * nbytes
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"{path}: short slab record")
    return unpack_3bit(data, int(header["n_metal_sites"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--composition", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--temperature-index", type=int, default=0)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()

    header = read_header(args.shard)
    print(json.dumps(header, indent=2))
    if args.trial is not None:
        codes = read_slab(
            args.shard,
            header,
            args.trial,
            args.composition,
            args.run,
            args.temperature_index,
        )
        elements = header["elements"]
        counts = Counter(elements[code] for code in codes)
        print("element_counts", dict(counts))
        print("first_40_codes", codes[:40])
    if args.sha256:
        digest = hashlib.sha256()
        with args.shard.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        print("sha256", digest.hexdigest())


if __name__ == "__main__":
    main()
