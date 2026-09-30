"""Dataset discovery and id_prop.csv parsing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import glob


@dataclass(frozen=True, slots=True)
class TargetRecord:
    identifier: str
    target: float
    path: Path


def read_targets(path: str | Path) -> list[tuple[str, float]]:
    path = Path(path)
    rows: list[tuple[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_number}: expected identifier,target")
            identifier = row[0].strip()
            try:
                target = float(row[1])
            except ValueError:
                # Permit one conventional header row.
                if not rows and line_number == 1:
                    continue
                raise ValueError(f"{path}:{line_number}: target is not numeric: {row[1]!r}")
            if not identifier:
                raise ValueError(f"{path}:{line_number}: empty identifier")
            rows.append((identifier, target))
    if not rows:
        raise ValueError(f"No target rows found in {path}")
    return rows


def _expand_input(entry: str | Path, extensions: set[str]) -> list[Path]:
    text = str(entry)
    path = Path(text)
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions
        )
    if any(char in text for char in "*?[]"):
        return sorted(
            Path(p) for p in glob.glob(text, recursive=True)
            if Path(p).is_file() and Path(p).suffix.lower() in extensions
        )
    if path.is_file():
        return [path]
    return []


def discover_structures(
    entries: str | Path | Iterable[str | Path],
    extensions: Iterable[str] = (".cif", ".vasp", ".poscar", ".xyz", ".extxyz"),
) -> dict[str, Path]:
    if isinstance(entries, (str, Path)):
        entries = [entries]
    ext = {e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in extensions}
    paths: list[Path] = []
    for entry in entries:
        paths.extend(_expand_input(entry, ext))
    if not paths:
        raise FileNotFoundError(f"No structure files found under: {list(entries)}")

    by_id: dict[str, Path] = {}
    for path in paths:
        key = path.stem.lower()
        if key in by_id and by_id[key].resolve() != path.resolve():
            raise ValueError(
                f"Duplicate case-insensitive structure identifier {path.stem!r}: "
                f"{by_id[key]} and {path}"
            )
        by_id[key] = path
    return by_id


def match_targets(
    targets: list[tuple[str, float]],
    structures: dict[str, Path],
    strict: bool = True,
) -> tuple[list[TargetRecord], list[str]]:
    records: list[TargetRecord] = []
    missing: list[str] = []
    for identifier, target in targets:
        key = Path(identifier).stem.lower()
        path = structures.get(key)
        if path is None:
            missing.append(identifier)
            continue
        records.append(TargetRecord(identifier=identifier, target=target, path=path))
    if strict and missing:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f" ... (+{len(missing)-10} more)"
        raise FileNotFoundError(
            f"Missing structure files for {len(missing)} target rows: {preview}{extra}"
        )
    if not records:
        raise ValueError("No target rows could be matched to structure files")
    return records, missing
