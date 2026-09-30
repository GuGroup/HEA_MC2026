"""Command-line interface for training, prediction, inspection, and annealing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from ase.io import read, write

from .config import load_config, write_json
from .data import discover_structures
from .lattice import MapperConfig, map_structure
from .mc import physical_atoms, run_annealing, save_annealing_outputs
from .model import CEModel, save_training_outputs, train_model
from .topology import ClusterSpec, build_topology, topology_signature


def _print_metric_summary(metrics: dict[str, Any], energy_unit: str = "eV") -> None:
    holdout = metrics.get("holdout")
    full = metrics.get("full_refit")
    if holdout:
        print(
            f"Holdout: MAE={holdout['mae']:.8g} {energy_unit}, "
            f"RMSE={holdout['rmse']:.8g} {energy_unit}, n={holdout['n']}"
        )
        for family, row in holdout.get("by_family", {}).items():
            print(f"  {family}: MAE={row['mae']:.8g} {energy_unit}, n={row['n']}")
    if full:
        print(
            f"Full refit: MAE={full['mae']:.8g} {energy_unit}, "
            f"RMSE={full['rmse']:.8g} {energy_unit}, n={full['n']}"
        )


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, predictions, metrics = train_model(config, progress=print)
    output = args.output or config.get("output", {}).get("directory", "ce_training_output")
    save_training_outputs(model, predictions, metrics, output)
    _print_metric_summary(metrics, model.energy_unit)
    print(f"Saved model and diagnostics to {Path(output).resolve()}")
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model = CEModel.load(args.model)
    structures = discover_structures(args.inputs, args.extensions)
    topology_cache = {}
    rows = []
    for _, path in sorted(structures.items()):
        mapped = map_structure(path, model.mapper_config)
        signature = topology_signature(mapped)
        topology = topology_cache.get(signature)
        if topology is None:
            topology = build_topology(mapped, model.cluster_spec)
            topology_cache[signature] = topology
        result = model.predict_mapped(mapped, topology)
        rows.append({"identifier": path.stem, "path": str(path), **result})
        print(f"{path.stem:30s} {result['prediction']:.10g} {model.energy_unit} ({mapped.kind})")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} predictions to {output.resolve()}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    cfg_data = load_config(args.config) if args.config else {}
    mapper_cfg = MapperConfig.from_dict(cfg_data.get("lattice", cfg_data))
    cluster_spec = ClusterSpec.from_dict(cfg_data.get("clusters")) if args.topology else None
    structures = discover_structures(args.inputs, args.extensions)
    output_dir = Path(args.write_mapped) if args.write_mapped else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, path in sorted(structures.items()):
        mapped = map_structure(path, mapper_cfg)
        row = {
            "identifier": path.stem,
            "path": str(path),
            "kind": mapped.kind,
            "n_input_atoms": mapped.metadata.get("source_n_atoms", len(mapped.atoms)),
            "n_metal": mapped.n_metal,
            "n_lattice": mapped.n_lattice,
            "n_vacancy": mapped.n_lattice - mapped.n_metal,
            "nearest_neighbor_A": mapped.nearest_neighbor,
            "occupied_layers": mapped.metadata.get("occupied_layers"),
            "vacancy_layers": mapped.metadata.get("vacancy_layers"),
            "layer_spacing_A": mapped.metadata.get("layer_spacing"),
        }
        if cluster_spec:
            topology = build_topology(mapped, cluster_spec)
            row["n_pairs"] = topology.n_pairs
            row["n_triplets"] = topology.n_triplets
            row["triplet_geometries"] = ";".join("-".join(map(str, g)) for g in topology.geometry_keys)
        rows.append(row)
        print(
            f"{path.stem:30s} kind={mapped.kind:8s} metal={mapped.n_metal:5d} "
            f"lattice={mapped.n_lattice:5d} d_nn={mapped.nearest_neighbor:.5f} Å"
        )
        if output_dir:
            write(output_dir / f"{path.stem}_mapped.extxyz", mapped.atoms)
            write_json(output_dir / f"{path.stem}_mapping.json", mapped.metadata)
    frame = pd.DataFrame(rows)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"Saved inspection table to {output.resolve()}")
    return 0


def command_anneal(args: argparse.Namespace) -> int:
    model = CEModel.load(args.model)
    schedule_data = load_config(args.schedule)
    simulation = dict(schedule_data.get("simulation", {})) if isinstance(schedule_data, dict) else {}
    mapped = map_structure(args.structure, model.mapper_config)
    repeat = tuple(args.repeat) if args.repeat else tuple(simulation.get("repeat", [1, 1, 1]))
    if repeat != (1, 1, 1):
        mapped = mapped.repeat(repeat)
    moves = dict(simulation.get("moves", {}))
    include_vacancies = bool(
        args.include_vacancies or moves.get("include_vacancies", False)
    )
    active_layers = moves.get("active_layers", "all")
    frozen_species = moves.get("frozen_species", [])
    record_every = float(
        args.record_every_sweeps
        if args.record_every_sweeps is not None
        else simulation.get("record_every_sweeps", 1.0)
    )
    random_seed = int(args.seed if args.seed is not None else simulation.get("random_seed", 42))
    vacuum = float(
        args.output_vacuum
        if args.output_vacuum is not None
        else simulation.get("output_vacuum_angstrom", 20.0)
    )

    topology = build_topology(mapped, model.cluster_spec)
    print(
        f"Annealing {mapped.kind}: {mapped.n_metal} metals / {mapped.n_lattice} lattice sites; "
        f"pairs={topology.n_pairs}, triplets={topology.n_triplets}"
    )
    result = run_annealing(
        model,
        mapped,
        schedule_data,
        topology=topology,
        include_vacancies=include_vacancies,
        active_layers=active_layers,
        frozen_species=frozen_species,
        record_every_sweeps=record_every,
        random_seed=random_seed,
    )
    output = save_annealing_outputs(
        args.output,
        model,
        mapped,
        result,
        vacuum_angstrom=vacuum,
        extra_summary={
            "repeat": list(repeat),
            "random_seed": random_seed,
            "include_vacancies": include_vacancies,
            "active_layers": active_layers,
            "frozen_species": list(frozen_species),
            "output_vacuum_angstrom": vacuum,
        },
    )
    print(
        f"Best target={result.best_total_energy / model.feature_spec.denominator(model.target_normalization, mapped.n_metal, mapped.n_lattice):.10g} "
        f"{model.energy_unit}; accepted={result.accepted_moves}/{result.attempted_moves}"
    )
    print(f"Saved annealing outputs to {output.resolve()}")
    return 0


def command_reconstruct(args: argparse.Namespace) -> int:
    run_directory = Path(args.run)
    checkpoint_path = run_directory / "anneal_checkpoint.npz"
    initial_path = run_directory / "initial_full_lattice.extxyz"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing annealing checkpoint: {checkpoint_path}")
    if not initial_path.is_file():
        raise FileNotFoundError(
            f"Missing initial mapped structure: {initial_path}. "
            "This run must be produced by fcc-ce 0.1.1 or newer."
        )

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if "swap_history" not in checkpoint:
            raise ValueError(
                "The checkpoint does not contain swap_history; rerun annealing with "
                "fcc-ce 0.1.1 or newer"
            )
        swap_history = np.asarray(checkpoint["swap_history"], dtype=np.int64).copy()
    if swap_history.ndim != 2 or swap_history.shape[1] != 2:
        raise ValueError("Checkpoint swap_history must have shape (n_steps, 2)")
    total_steps = int(swap_history.shape[0])

    if args.steps is not None:
        raw_steps = list(args.steps)
    elif args.every is not None:
        if args.every < 1:
            raise ValueError("--every must be a positive integer")
        raw_steps = list(range(0, total_steps + 1, int(args.every)))
        if not raw_steps or raw_steps[-1] != total_steps:
            raw_steps.append(total_steps)
    else:
        trace_path = run_directory / "anneal_trace.csv"
        if trace_path.is_file():
            raw_steps = pd.read_csv(trace_path, usecols=["step"])["step"].astype(int).tolist()
        else:
            raw_steps = [0, total_steps]

    resolved_steps: list[int] = []
    for step in raw_steps:
        resolved = total_steps if int(step) == -1 else int(step)
        if resolved < 0 or resolved > total_steps:
            raise ValueError(
                f"Requested step {step}; valid completed-step values are 0..{total_steps}, "
                "or -1 for final"
            )
        resolved_steps.append(resolved)
    resolved_steps = sorted(set(resolved_steps))
    if not resolved_steps:
        raise ValueError("No reconstruction steps were selected")

    initial_atoms = read(initial_path)
    initial_symbols = np.asarray(initial_atoms.get_chemical_symbols())
    if len(initial_symbols) == 0:
        raise ValueError("Initial mapped structure contains no sites")
    if "ce_site_index" not in initial_atoms.arrays:
        initial_atoms.set_array("ce_site_index", np.arange(len(initial_atoms), dtype=np.int32))

    non_null = swap_history[:, 0] >= 0
    if np.any((swap_history[:, 0] < 0) != (swap_history[:, 1] < 0)):
        raise ValueError("Checkpoint contains a partially null swap row")
    if np.any(swap_history[non_null] >= len(initial_atoms)):
        raise ValueError("Checkpoint contains a site index outside the initial structure")
    if np.any(swap_history[non_null] < 0):
        raise ValueError("Checkpoint contains an invalid negative site index")
    accepted_cumulative = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(non_null, dtype=np.int64)]
    )

    summary_path = run_directory / "anneal_summary.json"
    summary = load_config(summary_path) if summary_path.is_file() else {}
    vacancy_symbol = str(summary.get("vacancy_symbol", "X"))
    kind = str(summary.get("kind", initial_atoms.info.get("ce_kind", "bulk")))
    vacuum = float(
        args.output_vacuum
        if args.output_vacuum is not None
        else summary.get("output_vacuum_angstrom", 20.0)
    )

    output = Path(args.output) if args.output else run_directory / "reconstructed"
    output.mkdir(parents=True, exist_ok=True)
    width = max(8, len(str(total_steps)))
    symbols = initial_symbols.copy()
    current_step = 0
    manifest: list[dict[str, Any]] = []
    for target_step in resolved_steps:
        for history_index in range(current_step, target_step):
            site_a = int(swap_history[history_index, 0])
            site_b = int(swap_history[history_index, 1])
            if site_a >= 0:
                symbols[site_a], symbols[site_b] = symbols[site_b], symbols[site_a]
        current_step = target_step

        atoms = initial_atoms.copy()
        atoms.set_chemical_symbols(symbols.tolist())
        atoms.info["mc_step"] = int(target_step)
        atoms.info["accepted_moves"] = int(accepted_cumulative[target_step])
        full_name = f"step_{target_step:0{width}d}_full_lattice.extxyz"
        write(output / full_name, atoms)
        physical_name: str | None = None
        if args.physical:
            physical_name = f"step_{target_step:0{width}d}_physical.cif"
            write(
                output / physical_name,
                physical_atoms(atoms, vacancy_symbol, kind, vacuum),
            )
        manifest.append(
            {
                "step": target_step,
                "accepted_moves": int(accepted_cumulative[target_step]),
                "full_lattice_file": full_name,
                "physical_file": physical_name,
            }
        )

    pd.DataFrame(manifest).to_csv(output / "reconstructed_steps.csv", index=False)
    print(
        f"Reconstructed {len(resolved_steps)} state(s) from steps 0..{total_steps}; "
        f"saved to {output.resolve()}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcc-ce",
        description="Vacancy-aware fcc cluster expansion for bulk, (111), (100), and (110) slabs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit and save a cluster expansion")
    train.add_argument("--config", required=True, help="training YAML/JSON configuration")
    train.add_argument("--output", help="override output directory")
    train.set_defaults(func=command_train)

    predict = sub.add_parser("predict", help="predict one or more structures")
    predict.add_argument("--model", required=True, help="saved model directory")
    predict.add_argument("--inputs", nargs="+", required=True, help="files, directories, or globs")
    predict.add_argument("--output", default="predictions.csv")
    predict.add_argument(
        "--extensions",
        nargs="+",
        default=[".cif", ".vasp", ".poscar", ".xyz", ".extxyz"],
    )
    predict.set_defaults(func=command_predict)

    inspect = sub.add_parser("inspect", help="validate automatic lattice reconstruction")
    inspect.add_argument("--inputs", nargs="+", required=True)
    inspect.add_argument("--config", help="optional mapper/cluster YAML")
    inspect.add_argument("--topology", action="store_true", help="also count clusters")
    inspect.add_argument("--output", help="optional CSV report")
    inspect.add_argument("--write-mapped", help="directory for explicit-vacancy extxyz files")
    inspect.add_argument(
        "--extensions",
        nargs="+",
        default=[".cif", ".vasp", ".poscar", ".xyz", ".extxyz"],
    )
    inspect.set_defaults(func=command_inspect)

    anneal = sub.add_parser("anneal", help="run programmable canonical Monte Carlo")
    anneal.add_argument("--model", required=True)
    anneal.add_argument("--structure", required=True)
    anneal.add_argument("--schedule", required=True)
    anneal.add_argument("--output", default="anneal_output")
    anneal.add_argument("--repeat", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    anneal.add_argument("--seed", type=int)
    anneal.add_argument("--record-every-sweeps", type=float)
    anneal.add_argument("--include-vacancies", action="store_true")
    anneal.add_argument("--output-vacuum", type=float)
    anneal.set_defaults(func=command_anneal)

    reconstruct = sub.add_parser(
        "reconstruct",
        help="replay an annealing swap history and write selected structures",
    )
    reconstruct.add_argument("--run", required=True, help="annealing output directory")
    selection = reconstruct.add_mutually_exclusive_group()
    selection.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="completed MC steps to write; 0 is initial and -1 is final",
    )
    selection.add_argument(
        "--every",
        type=int,
        help="write every N completed MC steps, including initial and final",
    )
    reconstruct.add_argument("--output", help="output directory; default RUN/reconstructed")
    reconstruct.add_argument(
        "--physical",
        action="store_true",
        help="also write vacancy-free physical CIF files",
    )
    reconstruct.add_argument("--output-vacuum", type=float)
    reconstruct.set_defaults(func=command_reconstruct)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
