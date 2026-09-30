#!/usr/bin/env python3
"""Create per-worker UQMC config files from a prepared base config."""
from __future__ import annotations

import argparse
from pathlib import Path


def read_config(path: Path) -> list[tuple[str | None, str]]:
    rows: list[tuple[str | None, str]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rows.append((None, line))
            continue
        key, value = line.split("=", 1)
        rows.append((key.strip(), value.strip()))
    return rows


def write_config(rows: list[tuple[str | None, str]], output: Path, replacements: dict[str, str]) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for key, value in rows:
        if key is None:
            lines.append(value)
            continue
        if key in replacements:
            lines.append(f"{key} = {replacements[key]}")
            seen.add(key)
        else:
            lines.append(f"{key} = {value}")
    for key, value in replacements.items():
        if key not in seen:
            lines.append(f"{key} = {value}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="uq_config.ini")
    p.add_argument("--output-dir", default="shards")
    p.add_argument("--results-root", default="results")
    p.add_argument("--world-size", type=int, default=4)
    args = p.parse_args()

    base = Path(args.base_config).resolve()
    outdir = Path(args.output_dir).resolve()
    results_root = Path(args.results_root).resolve()
    rows = read_config(base)
    for idx in range(args.world_size):
        cfg = outdir / f"uq_config_shard_{idx:02d}.ini"
        replacements = {
            "output_dir": str(results_root / f"shard_{idx:02d}"),
            "trial_start": str(idx),
            "trial_stride": str(args.world_size),
        }
        write_config(rows, cfg, replacements)
        print(cfg)


if __name__ == "__main__":
    main()
