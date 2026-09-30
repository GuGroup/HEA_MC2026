"""YAML/JSON configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level configuration in {path} must be a mapping")
    return data


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
