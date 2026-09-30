#!/usr/bin/env python3
"""Prepare C++ input files for HEA UQ/CEMC.

This script is intentionally Python because it performs one-time translation from
fcc-ce's Python model format (.npz + metadata.json) into flat text files consumed
by the C++/MPI executable. The production UQ/CEMC loop then runs in C++.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.build import fcc111
from ase.io import write

DEFAULT_PROJECT_DIR = Path("/home/ktg0829/project/HEA/CEMC")
DEFAULT_COMPOSITION_CSV = DEFAULT_PROJECT_DIR / "composition_new.csv"
DEFAULT_ACTIVITY_JSON = DEFAULT_PROJECT_DIR / "orr_matched_activity.json"
DEFAULT_MODEL_DIR = DEFAULT_PROJECT_DIR / "CEMC/ce_training_output/model"
DEFAULT_SCHEDULE_YAML = Path("schedule.yaml")
DEFAULT_ACTIVITY_PYTHON = Path("predict_orr_activity_models.py")


def _parse_ml_index(value: int | str) -> int:
    text = str(value).strip()
    if text.upper().startswith("ML"):
        text = text[2:]
    return int(text)


def _load_fcc_ce():
    try:
        from fcc_ce.model import CEModel
        from fcc_ce.lattice import map_structure
        from fcc_ce.topology import build_topology
        return CEModel, map_structure, build_topology
    except Exception as exc:
        raise SystemExit(
            "Could not import fcc_ce. Activate the conda environment where fcc-ce is installed, "
            "for example: conda activate CEMC_2026-06-22\n"
            f"Import error: {exc}"
        ) from exc


def _write_values(handle, key: str, values: list[Any] | np.ndarray, per_line: int = 16) -> None:
    vals = list(np.asarray(values).ravel()) if not isinstance(values, list) else values
    handle.write(f"{key}")
    if len(vals) == 0:
        handle.write("\n")
        return
    for i, v in enumerate(vals):
        if i % per_line == 0:
            handle.write("\n")
        if isinstance(v, (np.floating, float)):
            handle.write(f" {float(v):.17g}")
        else:
            handle.write(f" {v}")
    handle.write("\n")


def _format_vector(values: list[Any] | np.ndarray, per_line: int = 16) -> str:
    vals = list(np.asarray(values).ravel()) if not isinstance(values, list) else values
    chunks = []
    for i in range(0, len(vals), per_line):
        part = vals[i:i + per_line]
        chunks.append(" ".join(f"{float(v):.17g}" if isinstance(v, (np.floating, float)) else str(v) for v in part))
    return "\n".join(chunks)


def _generate_template(path: Path, size: tuple[int, int, int], lattice_constant: float, vacuum: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = fcc111("Pt", size=size, a=lattice_constant, vacuum=vacuum, orthogonal=False)
    atoms.pbc = True
    write(path, atoms)


def _neighbor_activity_topology(mapped, topology):
    layer = np.asarray(mapped.layer_index, dtype=int)
    symbols = np.asarray(mapped.symbols)
    vacancy = mapped.vacancy_symbol
    metal_mask = symbols != vacancy
    top_layer = int(layer[metal_mask].max())
    sub_layer = top_layer - 1
    surface_sites = np.flatnonzero(metal_mask & (layer == top_layer)).astype(int)
    surface_sites = np.asarray(sorted(surface_sites.tolist()), dtype=np.int32)
    zone2: list[list[int]] = [[] for _ in surface_sites]
    zone3: list[list[int]] = [[] for _ in surface_sites]
    pos = {int(site): i for i, site in enumerate(surface_sites)}
    for i, j, sh in zip(topology.pair_i, topology.pair_j, topology.pair_shell):
        if int(sh) != 1:
            continue
        i = int(i); j = int(j)
        if i in pos and j != i:
            if layer[j] == top_layer:
                zone2[pos[i]].append(j)
            elif layer[j] == sub_layer:
                zone3[pos[i]].append(j)
        if j in pos and i != j:
            if layer[i] == top_layer:
                zone2[pos[j]].append(i)
            elif layer[i] == sub_layer:
                zone3[pos[j]].append(i)
    z2_ptr = [0]
    z2_idx: list[int] = []
    z3_ptr = [0]
    z3_idx: list[int] = []
    bad = []
    for site, a, b in zip(surface_sites, zone2, zone3):
        a2 = sorted(set(int(x) for x in a))
        b2 = sorted(set(int(x) for x in b))
        if len(a2) != 6 or len(b2) != 3:
            bad.append((int(site), len(a2), len(b2)))
        z2_idx.extend(a2); z2_ptr.append(len(z2_idx))
        z3_idx.extend(b2); z3_ptr.append(len(z3_idx))
    if bad:
        print("WARNING: Some fcc(111) top sites do not have 6 same-layer + 3 subsurface NN in the activity topology.", file=sys.stderr)
        print("First few:", bad[:10], file=sys.stderr)
    return surface_sites, np.asarray(z2_ptr, dtype=np.int32), np.asarray(z2_idx, dtype=np.int32), np.asarray(z3_ptr, dtype=np.int32), np.asarray(z3_idx, dtype=np.int32)


def export_ce(args) -> tuple[int, int, int]:
    CEModel, map_structure, build_topology = _load_fcc_ce()
    model = CEModel.load(args.model)

    template = Path(args.template_cif) if args.template_cif else Path(args.workdir) / "template_fcc111_10x10x10.cif"
    if not template.exists():
        sx, sy, sz = args.slab_size
        _generate_template(template, (sx, sy, sz), args.lattice_constant, args.vacuum)

    mapped = map_structure(template, model.mapper_config)
    topology = build_topology(mapped, model.cluster_spec)
    tables = model.coefficient_tables
    species = list(model.feature_spec.species)
    atomic_numbers = []
    zmap = {"Ir": 77, "Pd": 46, "Pt": 78, "Rh": 45, "Ru": 44, model.feature_spec.vacancy_symbol: 0, "X": 0}
    for s in species:
        atomic_numbers.append(zmap.get(s, 0))
    symbols = np.asarray(mapped.symbols)
    metal_sites = np.flatnonzero(symbols != mapped.vacancy_symbol).astype(np.int32)
    active_sites = metal_sites.copy()
    surface_sites, z2_ptr, z2_idx, z3_ptr, z3_idx = _neighbor_activity_topology(mapped, topology)

    geom_ids = np.array([model.feature_spec.geometry_to_id[tuple(g)] for g in topology.triplet_geometry], dtype=np.int32)
    pair = np.asarray(tables.pair, dtype=float)
    triplet = np.asarray(tables.triplet, dtype=float)
    if triplet.size == 0:
        triplet = np.zeros((0, len(species), len(species), len(species)), dtype=float)

    out = Path(args.ce_export)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("FCC_CE_EXPORT_V1\n")
        f.write(f"n_species {len(species)}\n")
        f.write("species " + " ".join(species) + "\n")
        f.write("atomic_numbers " + " ".join(str(x) for x in atomic_numbers) + "\n")
        f.write(f"vacancy_code {model.feature_spec.vacancy_code}\n")
        f.write(f"n_sites {len(mapped.atoms)}\n")
        f.write(f"n_metal_sites {len(metal_sites)}\n")
        _write_values(f, "metal_sites", metal_sites)
        f.write(f"n_active_sites {len(active_sites)}\n")
        _write_values(f, "active_sites", active_sites)
        f.write(f"n_surface_sites {len(surface_sites)}\n")
        _write_values(f, "surface_sites", surface_sites)
        f.write(f"zero {float(tables.zero):.17g}\n")
        _write_values(f, "point", np.asarray(tables.point, dtype=float))
        f.write(f"pair_shape {pair.shape[0]} {pair.shape[1]} {pair.shape[2]}\n")
        f.write(_format_vector(pair.ravel()) + "\n")
        f.write(f"n_pairs {topology.n_pairs}\n")
        _write_values(f, "pair_i", np.asarray(topology.pair_i, dtype=np.int32))
        _write_values(f, "pair_j", np.asarray(topology.pair_j, dtype=np.int32))
        _write_values(f, "pair_shell", np.asarray(topology.pair_shell, dtype=np.int32))
        f.write(f"triplet_shape {triplet.shape[0]} {triplet.shape[1]} {triplet.shape[2]} {triplet.shape[3]}\n")
        if triplet.size:
            f.write(_format_vector(triplet.ravel()) + "\n")
        f.write(f"n_triplets {topology.n_triplets}\n")
        _write_values(f, "triplet_i", np.asarray(topology.triplet_i, dtype=np.int32))
        _write_values(f, "triplet_j", np.asarray(topology.triplet_j, dtype=np.int32))
        _write_values(f, "triplet_k", np.asarray(topology.triplet_k, dtype=np.int32))
        _write_values(f, "triplet_gid", geom_ids)
        _write_values(f, "site_pair_ptr", np.asarray(topology.site_pair_ptr, dtype=np.int32))
        f.write(f"site_pair_indices {len(topology.site_pair_indices)}\n")
        f.write(_format_vector(np.asarray(topology.site_pair_indices, dtype=np.int32)) + "\n")
        _write_values(f, "site_triplet_ptr", np.asarray(topology.site_triplet_ptr, dtype=np.int32))
        f.write(f"site_triplet_indices {len(topology.site_triplet_indices)}\n")
        f.write(_format_vector(np.asarray(topology.site_triplet_indices, dtype=np.int32)) + "\n")
    print(f"Wrote CE export: {out}")
    print(f"Mapped slab kind={mapped.kind}, n_sites={len(mapped.atoms)}, n_metal={len(metal_sites)}, n_surface={len(surface_sites)}, n_pairs={topology.n_pairs}, n_triplets={topology.n_triplets}")

    return len(active_sites), len(surface_sites), len(mapped.atoms)


def _schedule_temperature_grid(spec: str) -> list[float]:
    # Accept comma list or start:stop:step, inclusive when step sign points to stop.
    if ":" not in spec:
        return [float(x) for x in spec.split(",") if x.strip()]
    parts = [float(x) for x in spec.split(":")]
    if len(parts) != 3:
        raise ValueError("temperature grid must be comma-list or start:stop:step")
    start, stop, step = parts
    vals = []
    x = start
    if step == 0:
        raise ValueError("temperature grid step cannot be zero")
    if step > 0:
        while x <= stop + 1e-9:
            vals.append(x); x += step
    else:
        while x >= stop - 1e-9:
            vals.append(x); x += step
    return vals


def export_schedule(args, n_active_sites: int) -> None:
    """Export the *unchanged* CEMC schedule plus target snapshot temperatures.

    This intentionally does not replace the ramp stop temperature with the grid
    temperature. The C++ code runs the original schedule once and extracts
    snapshots when the trajectory crosses 2000, 1900, ..., 300 K.
    """
    with open(args.schedule_yaml) as f:
        data = yaml.safe_load(f)
    segments = data.get("schedule", data.get("segments", data if isinstance(data, list) else None))
    if not isinstance(segments, list) or not segments:
        raise ValueError("schedule yaml must contain a schedule list")
    targets = _schedule_temperature_grid(args.temperature_grid)
    out = Path(args.schedule_export)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("SCHEDULE_SNAPSHOTS_V1\n")
        f.write(f"n_targets {len(targets)}\n")
        f.write("target_temperatures " + " ".join(f"{float(t):.12g}" for t in targets) + "\n")
        f.write(f"n_segments {len(segments)}\n")
        for seg in segments:
            kind = str(seg.get("type", seg.get("kind", "hold"))).lower()
            steps = seg.get("steps")
            if steps is None:
                sweeps = seg.get("sweeps")
                if sweeps is None:
                    raise ValueError(f"schedule segment {seg} must contain steps or sweeps")
                steps = int(round(float(sweeps) * n_active_sites))
            else:
                steps = int(steps)
            if kind in {"hold", "burn_in", "burn-in", "equilibrate"}:
                T = float(seg.get("temperature", seg.get("T", seg.get("start", 300.0))))
                profile, start, stop = 0, T, T
            elif kind in {"ramp", "anneal", "cool", "heat"}:
                start = float(seg.get("start", seg.get("start_temperature", seg.get("T_start", 2000.0))))
                stop = float(seg.get("stop", seg.get("stop_temperature", seg.get("T_stop", 300.0))))
                pname = str(seg.get("profile", seg.get("cooling", "linear"))).lower()
                profile = 2 if pname in {"exponential", "exp", "geometric"} else 1
            else:
                raise ValueError(f"Unsupported schedule segment type: {kind}")
            f.write(f"segment {profile} {start:.12g} {stop:.12g} {int(steps)}\n")
    print(f"Wrote schedule export without changing schedule endpoints: {out}")


def write_activity_model(args, ce_export_path: Path) -> None:
    # Append neighbor topology from CE export is not convenient to read here, so the topology
    # is written by export_ce into a sidecar numpy dictionary first? Instead, recompute by loading
    # the CE export text minimally and reading surface/zone arrays from temporary JSON.
    # This function writes only coefficient template. The CE export already contains surface_sites;
    # activity_model must contain zone topology. Therefore export_ce writes a JSON sidecar now.
    pass


def export_activity_model(args) -> None:
    sidecar = Path(args.workdir) / "activity_topology.json"
    if not sidecar.exists():
        raise SystemExit(f"Missing {sidecar}. Run CE export first.")
    topo = json.loads(sidecar.read_text())

    out = Path(args.activity_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    coeffs = None
    if args.activity_python:
        coeffs = try_import_activity_python(Path(args.activity_python))
    if coeffs is None and out.exists():
        text = out.read_text().strip()
        if text.startswith("ACTIVITY_MODEL_OH_V1"):
            # Keep existing coefficients but rewrite topology to match current slab.
            coeffs = parse_existing_activity_coefficients(text)
    if coeffs is None:
        coeffs = {
            "elements": ["Ir", "Pd", "Pt", "Rh", "Ru"],
            "intercept": 0.0,
            "zone1": [0.0]*5,
            "zone2": [0.0]*5,
            "zone3": [0.0]*5,
            "e_opt": 1.1,
            "activity_temperature": 300.0,
        }
        print("WARNING: No OH linear-model coefficients were found. Wrote a placeholder activity_model_oh.txt.", file=sys.stderr)
        print("         Fill zone1/zone2/zone3 from Get_activities.py or use --activity-python with get_cpp_oh_model().", file=sys.stderr)

    with out.open("w") as f:
        f.write("ACTIVITY_MODEL_OH_V1\n")
        f.write("elements " + " ".join(coeffs["elements"]) + "\n")
        f.write(f"intercept {float(coeffs.get('intercept', 0.0)):.17g}\n")
        for key in ["zone1", "zone2", "zone3"]:
            vals = coeffs[key]
            if len(vals) != 5:
                raise ValueError(f"{key} must contain 5 coefficients")
            f.write(key + " " + " ".join(f"{float(x):.17g}" for x in vals) + "\n")
        f.write(f"e_opt {float(coeffs.get('e_opt', 1.1)):.17g}\n")
        f.write(f"activity_temperature {float(coeffs.get('activity_temperature', 300.0)):.17g}\n")
        f.write("zone2_ptr " + " ".join(str(int(x)) for x in topo["zone2_ptr"]) + "\n")
        f.write(f"zone2_indices {len(topo['zone2_indices'])}\n")
        f.write(" ".join(str(int(x)) for x in topo["zone2_indices"]) + "\n")
        f.write("zone3_ptr " + " ".join(str(int(x)) for x in topo["zone3_ptr"]) + "\n")
        f.write(f"zone3_indices {len(topo['zone3_indices'])}\n")
        f.write(" ".join(str(int(x)) for x in topo["zone3_indices"]) + "\n")
    print(f"Wrote activity model: {out}")


def parse_existing_activity_coefficients(text: str) -> dict[str, Any]:
    toks = text.split()
    if not toks or toks[0] != "ACTIVITY_MODEL_OH_V1":
        raise ValueError("not an activity model")
    i = 1
    coeffs: dict[str, Any] = {}
    while i < len(toks):
        key = toks[i]; i += 1
        if key == "elements": coeffs[key] = toks[i:i+5]; i += 5
        elif key in {"zone1", "zone2", "zone3"}: coeffs[key] = [float(x) for x in toks[i:i+5]]; i += 5
        elif key in {"intercept", "e_opt", "activity_temperature"}: coeffs[key] = float(toks[i]); i += 1
        elif key in {"zone2_ptr", "zone3_ptr"}:
            # Stop reading old topology. It will be replaced.
            break
        elif key in {"zone2_indices", "zone3_indices"}:
            n = int(toks[i]); i += 1 + n
        else:
            break
    for k in ["elements", "zone1", "zone2", "zone3"]:
        if k not in coeffs:
            raise ValueError(f"existing activity model missing {k}")
    coeffs.setdefault("intercept", 0.0)
    coeffs.setdefault("e_opt", 1.1)
    coeffs.setdefault("activity_temperature", 300.0)
    return coeffs


def try_import_activity_python(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"WARNING: activity python file not found: {path}", file=sys.stderr)
        return None

    # Get_activities.py has top-level file I/O in the provided version, so importing
    # it directly would execute the workflow. Parse its coefficient literals instead.
    if path.name == "Get_activities.py":
        parsed = parse_get_activities_coefficients(path)
        if parsed is not None:
            return parsed

    spec = importlib.util.spec_from_file_location("activity_user_model", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"WARNING: could not import {path}: {exc}. Trying literal parser or using existing/template activity_model_oh.txt.", file=sys.stderr)
        return parse_get_activities_coefficients(path)

    if hasattr(mod, "get_cpp_oh_model"):
        data = mod.get_cpp_oh_model()
        required = ["elements", "zone1", "zone2", "zone3"]
        if not all(k in data for k in required):
            raise ValueError("get_cpp_oh_model() must return elements, zone1, zone2, zone3")
        data.setdefault("e_opt", 1.1)
        data.setdefault("activity_temperature", 298.0)
        return data

    # Provided predict_orr_activity_models.py exposes OH_PARAMS and ELEMENTS.
    if hasattr(mod, "OH_PARAMS") and hasattr(mod, "ELEMENTS"):
        elements = list(mod.ELEMENTS)
        oh = mod.OH_PARAMS
        return {
            "elements": elements,
            "intercept": 0.0,
            "zone1": [float(oh[1][e]) for e in elements],
            "zone2": [float(oh[2][e]) for e in elements],
            "zone3": [float(oh[3][e]) for e in elements],
            "e_opt": 1.1,
            "activity_temperature": 298.0,
        }

    parsed = parse_get_activities_coefficients(path)
    if parsed is not None:
        return parsed
    print(
        f"WARNING: {path} has no get_cpp_oh_model() or OH_PARAMS. "
        "Using existing/template activity_model_oh.txt instead.",
        file=sys.stderr,
    )
    return None


def parse_get_activities_coefficients(path: Path) -> dict[str, Any] | None:
    """Parse an_map and coeff literals from Get_activities.py without executing it."""
    import ast
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"an_map", "coeff"}:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    if "an_map" not in values or "coeff" not in values:
        return None
    z_to_sym = {77: "Ir", 46: "Pd", 78: "Pt", 45: "Rh", 44: "Ru"}
    elements = [z_to_sym[int(z)] for z in values["an_map"]]
    coeff = [float(x) for x in values["coeff"]]
    if len(elements) != 5 or len(coeff) != 15:
        return None
    return {
        "elements": elements,
        "intercept": 0.0,
        "zone1": coeff[0:5],
        "zone2": coeff[5:10],
        "zone3": coeff[10:15],
        "e_opt": 1.1,
        "activity_temperature": 298.0,
    }


def write_activity_topology_sidecar(args) -> None:
    # Reload model/template to write the topology sidecar used by export_activity_model.
    CEModel, map_structure, build_topology = _load_fcc_ce()
    model = CEModel.load(args.model)
    template = Path(args.template_cif) if args.template_cif else Path(args.workdir) / "template_fcc111_10x10x10.cif"
    mapped = map_structure(template, model.mapper_config)
    topology = build_topology(mapped, model.cluster_spec)
    surface_sites, z2_ptr, z2_idx, z3_ptr, z3_idx = _neighbor_activity_topology(mapped, topology)
    sidecar = Path(args.workdir) / "activity_topology.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "surface_sites": [int(x) for x in surface_sites],
        "zone2_ptr": [int(x) for x in z2_ptr],
        "zone2_indices": [int(x) for x in z2_idx],
        "zone3_ptr": [int(x) for x in z3_ptr],
        "zone3_indices": [int(x) for x in z3_idx],
    }, indent=2))


def write_config(args) -> None:
    cfg_path = Path(args.config_out)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# C++ UQ/CEMC configuration generated by prepare_inputs.py
ce_export = {Path(args.ce_export).resolve()}
schedule_export = {Path(args.schedule_export).resolve()}
activity_model = {Path(args.activity_model).resolve()}
composition_csv = {Path(args.composition_csv).resolve()}
experimental_activity_csv = {Path(args.activity_csv).resolve()}
output_dir = {Path(args.output_dir).resolve()}
ml_index = {args.ml}
n_trials = {args.n_trials}
n_runs = {args.n_runs}
max_compositions = {args.max_compositions}
trial_start = {args.trial_start}
trial_stride = {args.trial_stride}
random_seed = {args.random_seed}
comp_error_mean = {args.comp_error_mean}
comp_error_sigma = {args.comp_error_sigma}
comp_error_units = {args.comp_error_units}
be_error_mean = {args.be_error_mean}
be_error_sigma = {args.be_error_sigma}
write_structures = {str(args.write_structures).lower()}
write_random_structures = {str(args.write_random_structures).lower()}
write_random_seeds = {str(args.write_random_seeds).lower()}
write_packed_atoms = {str(args.write_packed_atoms).lower()}
write_predicted_activities = {str(args.write_predicted_activities).lower()}
write_all_snapshot_activities = false
write_trial_zarr_summaries = false
summary_every_trials = {args.summary_every_trials}
predicted_better_is_higher = true
experimental_better_is_lower = true
"""
    cfg_path.write_text(text)
    print(f"Wrote C++ config: {cfg_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare flat files for C++/MPI UQ CEMC")
    p.add_argument("--workdir", default="uq_ml1_debug_v10")
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--schedule-yaml", default=str(DEFAULT_SCHEDULE_YAML))
    p.add_argument("--composition-csv", default="", help=f"Default: {DEFAULT_COMPOSITION_CSV}")
    p.add_argument("--activity-csv", default="", help=f"Default: {DEFAULT_ACTIVITY_JSON}")
    p.add_argument("--activity-python", default=str(DEFAULT_ACTIVITY_PYTHON))
    p.add_argument("--template-cif", default="", help="Optional pre-existing fcc111 10x10x10 CIF. If absent, ASE generates one.")
    p.add_argument("--slab-size", type=int, nargs=3, default=[10, 10, 10])
    p.add_argument("--lattice-constant", type=float, default=3.9085046290)
    p.add_argument("--vacuum", type=float, default=20.0)
    p.add_argument("--temperature-grid", default="2000:300:-100")
    p.add_argument("--ml", default="1")
    p.add_argument("--n-trials", type=int, default=2)
    p.add_argument("--n-runs", type=int, default=2)
    p.add_argument("--max-compositions", type=int, default=8, help="Debug default. Use -1 for all 1400.")
    p.add_argument("--trial-start", type=int, default=0, help="First global UQ trial processed by this config.")
    p.add_argument("--trial-stride", type=int, default=1, help="Process trial_start, trial_start+trial_stride, ...")
    p.add_argument("--random-seed", type=int, default=20260706)
    p.add_argument("--comp-error-mean", type=float, default=0.000347)
    p.add_argument("--comp-error-sigma", type=float, default=0.046858)
    p.add_argument("--comp-error-units", choices=["fraction", "percent", "at_percent"], default="fraction")
    p.add_argument("--be-error-mean", type=float, default=0.04233)
    p.add_argument("--be-error-sigma", type=float, default=0.2604)
    p.add_argument("--write-structures", action=argparse.BooleanOptionalAction, default=False, help="Write full CEMC atomic-number lists. Default false for production to avoid huge JSONL I/O; snapshots can be reconstructed from seeds.")
    p.add_argument("--write-random-structures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--write-random-seeds", action=argparse.BooleanOptionalAction, default=True, help="Record seeds/counts needed to reconstruct random baseline slabs without storing them.")
    p.add_argument("--write-packed-atoms", action=argparse.BooleanOptionalAction, default=False, help="Write compact 3-bit random/CEMC atomic arrays in template ASE order.")
    p.add_argument("--write-predicted-activities", action=argparse.BooleanOptionalAction, default=True, help="Write selected-temperature composition-level predicted activity CSV used for activity maps and diagnostics.")
    p.add_argument("--summary-every-trials", type=int, default=0, help="0: write uncertainty/plot summaries only once at the end; N: also update every N completed trials.")
    args = p.parse_args()
    args.ml = _parse_ml_index(args.ml)
    if not args.composition_csv:
        args.composition_csv = str(DEFAULT_COMPOSITION_CSV)
    if not args.activity_csv:
        args.activity_csv = str(DEFAULT_ACTIVITY_JSON)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    args.ce_export = str(workdir / "ce_export.txt")
    args.schedule_export = str(workdir / "schedule_snapshots.txt")
    args.activity_model = str(workdir / "activity_model_oh.txt")
    args.config_out = str(workdir / "uq_config.ini")
    args.output_dir = str(workdir / "results")

    n_active, _, _ = export_ce(args)
    write_activity_topology_sidecar(args)
    export_schedule(args, n_active)
    export_activity_model(args)
    write_config(args)


if __name__ == "__main__":
    main()
