"""Fast canonical Metropolis annealing with local CE energy updates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write
from numba import njit

from .config import write_json
from .lattice import MappedStructure
from .model import CEModel
from .topology import Topology, build_topology


BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass
class CompiledSchedule:
    profiles: np.ndarray  # 0 hold, 1 linear, 2 exponential
    start_temperatures: np.ndarray
    stop_temperatures: np.ndarray
    steps: np.ndarray
    labels: list[str]

    @property
    def total_steps(self) -> int:
        return int(np.sum(self.steps))

    @property
    def boundaries(self) -> np.ndarray:
        return np.cumsum(self.steps)


@dataclass
class MCResult:
    initial_occupations: np.ndarray
    final_occupations: np.ndarray
    best_occupations: np.ndarray
    swap_history: np.ndarray
    trace: pd.DataFrame
    final_total_energy: float
    best_total_energy: float
    best_step: int
    attempted_moves: int
    accepted_moves: int


def _value(segment: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in segment:
            return segment[name]
    return default


def compile_schedule(schedule_data: Any, n_active_sites: int) -> CompiledSchedule:
    if isinstance(schedule_data, dict):
        segments = schedule_data.get("schedule", schedule_data.get("segments"))
    else:
        segments = schedule_data
    if not isinstance(segments, list) or not segments:
        raise ValueError("Schedule must be a non-empty list or contain a 'schedule' list")
    if n_active_sites < 2:
        raise ValueError("At least two active sites are required")

    profiles: list[int] = []
    starts: list[float] = []
    stops: list[float] = []
    steps: list[int] = []
    labels: list[str] = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise ValueError(f"Schedule segment {index} must be a mapping")
        kind = str(raw.get("type", raw.get("kind", "hold"))).lower()
        label = str(raw.get("label", f"segment_{index}_{kind}"))
        explicit_steps = raw.get("steps")
        sweeps = raw.get("sweeps")
        if explicit_steps is None and sweeps is None:
            raise ValueError(f"Schedule segment {label!r} requires steps or sweeps")
        n_steps = int(explicit_steps if explicit_steps is not None else round(float(sweeps) * n_active_sites))
        if n_steps < 1:
            raise ValueError(f"Schedule segment {label!r} has no trial steps")

        if kind in {"hold", "burn_in", "burn-in", "equilibrate"}:
            temperature = float(_value(raw, ["temperature", "T", "start"], None))
            if not np.isfinite(temperature) or temperature < 0:
                raise ValueError(f"Invalid hold temperature in segment {label!r}")
            profile = 0
            start = stop = temperature
        elif kind in {"ramp", "anneal", "cool", "heat"}:
            start = float(
                _value(raw, ["start", "start_temperature", "T_start", "temperature_start"], None)
            )
            stop = float(
                _value(raw, ["stop", "end", "stop_temperature", "T_stop", "temperature_stop"], None)
            )
            if not np.isfinite(start) or not np.isfinite(stop) or start < 0 or stop < 0:
                raise ValueError(f"Invalid ramp temperatures in segment {label!r}")
            profile_name = str(raw.get("profile", raw.get("cooling", "linear"))).lower()
            if profile_name == "linear":
                profile = 1
            elif profile_name in {"exponential", "exp", "geometric"}:
                if start <= 0 or stop <= 0:
                    raise ValueError("Exponential ramps require positive endpoint temperatures")
                profile = 2
            else:
                raise ValueError(f"Unknown ramp profile {profile_name!r}")
        else:
            raise ValueError(f"Unknown schedule segment type {kind!r}")

        profiles.append(profile)
        starts.append(start)
        stops.append(stop)
        steps.append(n_steps)
        labels.append(label)

    return CompiledSchedule(
        profiles=np.asarray(profiles, dtype=np.int8),
        start_temperatures=np.asarray(starts, dtype=float),
        stop_temperatures=np.asarray(stops, dtype=float),
        steps=np.asarray(steps, dtype=np.int64),
        labels=labels,
    )


@njit(cache=True)
def _total_energy(
    occupations: np.ndarray,
    vacancy_code: int,
    zero_coefficient: float,
    point_coefficients: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_shell: np.ndarray,
    pair_coefficients: np.ndarray,
    triplet_i: np.ndarray,
    triplet_j: np.ndarray,
    triplet_k: np.ndarray,
    triplet_geometry: np.ndarray,
    triplet_coefficients: np.ndarray,
) -> float:
    n_metal = 0
    energy = 0.0
    for code in occupations:
        if code != vacancy_code:
            n_metal += 1
        energy += point_coefficients[code]
    energy += zero_coefficient * n_metal
    for p in range(pair_i.size):
        energy += pair_coefficients[
            pair_shell[p], occupations[pair_i[p]], occupations[pair_j[p]]
        ]
    for t in range(triplet_i.size):
        energy += triplet_coefficients[
            triplet_geometry[t],
            occupations[triplet_i[t]],
            occupations[triplet_j[t]],
            occupations[triplet_k[t]],
        ]
    return energy


@njit(cache=True)
def _swapped_code(index: int, site_a: int, site_b: int, code_a: int, code_b: int, current: int) -> int:
    if index == site_a:
        return code_b
    if index == site_b:
        return code_a
    return current


@njit(cache=True)
def _swap_delta(
    site_a: int,
    site_b: int,
    occupations: np.ndarray,
    point_coefficients: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_shell: np.ndarray,
    pair_coefficients: np.ndarray,
    triplet_i: np.ndarray,
    triplet_j: np.ndarray,
    triplet_k: np.ndarray,
    triplet_geometry: np.ndarray,
    triplet_coefficients: np.ndarray,
    site_pair_ptr: np.ndarray,
    site_pair_indices: np.ndarray,
    site_triplet_ptr: np.ndarray,
    site_triplet_indices: np.ndarray,
    pair_marks: np.ndarray,
    triplet_marks: np.ndarray,
    token: int,
) -> float:
    code_a = int(occupations[site_a])
    code_b = int(occupations[site_b])
    if code_a == code_b:
        return 0.0
    delta = (
        point_coefficients[code_b]
        + point_coefficients[code_a]
        - point_coefficients[code_a]
        - point_coefficients[code_b]
    )
    # The point contribution cancels for a swap, but keeping the expression
    # explicit documents the canonical-composition invariant.

    for active in (site_a, site_b):
        for ptr in range(site_pair_ptr[active], site_pair_ptr[active + 1]):
            p = int(site_pair_indices[ptr])
            if pair_marks[p] == token:
                continue
            pair_marks[p] = token
            left = int(pair_i[p])
            right = int(pair_j[p])
            old_left = int(occupations[left])
            old_right = int(occupations[right])
            new_left = _swapped_code(left, site_a, site_b, code_a, code_b, old_left)
            new_right = _swapped_code(right, site_a, site_b, code_a, code_b, old_right)
            shell = int(pair_shell[p])
            delta += (
                pair_coefficients[shell, new_left, new_right]
                - pair_coefficients[shell, old_left, old_right]
            )

    for active in (site_a, site_b):
        for ptr in range(site_triplet_ptr[active], site_triplet_ptr[active + 1]):
            t = int(site_triplet_indices[ptr])
            if triplet_marks[t] == token:
                continue
            triplet_marks[t] = token
            i = int(triplet_i[t])
            j = int(triplet_j[t])
            k = int(triplet_k[t])
            old_i = int(occupations[i])
            old_j = int(occupations[j])
            old_k = int(occupations[k])
            new_i = _swapped_code(i, site_a, site_b, code_a, code_b, old_i)
            new_j = _swapped_code(j, site_a, site_b, code_a, code_b, old_j)
            new_k = _swapped_code(k, site_a, site_b, code_a, code_b, old_k)
            geometry = int(triplet_geometry[t])
            delta += (
                triplet_coefficients[geometry, new_i, new_j, new_k]
                - triplet_coefficients[geometry, old_i, old_j, old_k]
            )
    return delta


@njit(cache=True)
def _temperature(profile: int, start: float, stop: float, local_step: int, n_steps: int) -> float:
    if profile == 0 or n_steps <= 1:
        return start
    fraction = local_step / (n_steps - 1.0)
    if profile == 1:
        return start + fraction * (stop - start)
    return start * math.exp(fraction * math.log(stop / start))


@njit(cache=True)
def _run_mc_kernel(
    occupations_initial: np.ndarray,
    active_sites: np.ndarray,
    vacancy_code: int,
    zero_coefficient: float,
    point_coefficients: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_shell: np.ndarray,
    pair_coefficients: np.ndarray,
    triplet_i: np.ndarray,
    triplet_j: np.ndarray,
    triplet_k: np.ndarray,
    triplet_geometry: np.ndarray,
    triplet_coefficients: np.ndarray,
    site_pair_ptr: np.ndarray,
    site_pair_indices: np.ndarray,
    site_triplet_ptr: np.ndarray,
    site_triplet_indices: np.ndarray,
    schedule_profiles: np.ndarray,
    schedule_start: np.ndarray,
    schedule_stop: np.ndarray,
    schedule_steps: np.ndarray,
    record_interval: int,
    random_seed: int,
    boltzmann_constant: float,
):
    np.random.seed(random_seed)
    occupations = occupations_initial.copy()
    best_occupations = occupations.copy()
    energy = _total_energy(
        occupations,
        vacancy_code,
        zero_coefficient,
        point_coefficients,
        pair_i,
        pair_j,
        pair_shell,
        pair_coefficients,
        triplet_i,
        triplet_j,
        triplet_k,
        triplet_geometry,
        triplet_coefficients,
    )
    best_energy = energy
    best_step = 0
    total_steps = int(np.sum(schedule_steps))
    # One row per MC step. [-1, -1] means no swap was actually performed
    # at that step (a rejected proposal or no valid proposal).
    swap_history = np.full((total_steps, 2), -1, dtype=np.int32)
    max_records = total_steps // record_interval + len(schedule_steps) + 3
    record_step = np.empty(max_records, dtype=np.int64)
    record_temperature = np.empty(max_records, dtype=np.float64)
    record_energy = np.empty(max_records, dtype=np.float64)
    record_acceptance = np.empty(max_records, dtype=np.float64)
    record_segment = np.empty(max_records, dtype=np.int32)
    record_accepted_total = np.empty(max_records, dtype=np.int64)

    record_count = 1
    record_step[0] = 0
    record_temperature[0] = schedule_start[0]
    record_energy[0] = energy
    record_acceptance[0] = 0.0
    record_segment[0] = 0
    record_accepted_total[0] = 0

    pair_marks = np.zeros(pair_i.size, dtype=np.int64)
    triplet_marks = np.zeros(triplet_i.size, dtype=np.int64)
    global_step = 0
    accepted_total = 0
    accepted_window = 0
    attempted_window = 0
    last_record_step = 0
    last_temperature = schedule_start[0]

    for segment in range(schedule_steps.size):
        n_steps = int(schedule_steps[segment])
        for local_step in range(n_steps):
            temperature = _temperature(
                int(schedule_profiles[segment]),
                float(schedule_start[segment]),
                float(schedule_stop[segment]),
                local_step,
                n_steps,
            )
            last_temperature = temperature
            site_a = -1
            site_b = -1
            for _ in range(64):
                site_a = int(active_sites[np.random.randint(active_sites.size)])
                site_b = int(active_sites[np.random.randint(active_sites.size)])
                if site_a != site_b and occupations[site_a] != occupations[site_b]:
                    break
            if site_a != site_b and occupations[site_a] != occupations[site_b]:
                token = global_step + 1
                delta = _swap_delta(
                    site_a,
                    site_b,
                    occupations,
                    point_coefficients,
                    pair_i,
                    pair_j,
                    pair_shell,
                    pair_coefficients,
                    triplet_i,
                    triplet_j,
                    triplet_k,
                    triplet_geometry,
                    triplet_coefficients,
                    site_pair_ptr,
                    site_pair_indices,
                    site_triplet_ptr,
                    site_triplet_indices,
                    pair_marks,
                    triplet_marks,
                    token,
                )
                accept = delta <= 0.0
                if not accept and temperature > 0.0:
                    accept = np.random.random() < math.exp(-delta / (boltzmann_constant * temperature))
                attempted_window += 1
                if accept:
                    temp_code = occupations[site_a]
                    occupations[site_a] = occupations[site_b]
                    occupations[site_b] = temp_code
                    swap_history[global_step, 0] = site_a
                    swap_history[global_step, 1] = site_b
                    energy += delta
                    accepted_total += 1
                    accepted_window += 1
                    if energy < best_energy:
                        best_energy = energy
                        best_step = global_step + 1
                        best_occupations[:] = occupations
            global_step += 1

            at_interval = global_step % record_interval == 0
            at_segment_end = local_step == n_steps - 1
            if at_interval or at_segment_end:
                record_step[record_count] = global_step
                record_temperature[record_count] = temperature
                record_energy[record_count] = energy
                record_acceptance[record_count] = (
                    accepted_window / attempted_window if attempted_window > 0 else 0.0
                )
                record_segment[record_count] = segment
                record_accepted_total[record_count] = accepted_total
                record_count += 1
                accepted_window = 0
                attempted_window = 0
                last_record_step = global_step

    if last_record_step != global_step:
        record_step[record_count] = global_step
        record_temperature[record_count] = last_temperature
        record_energy[record_count] = energy
        record_acceptance[record_count] = (
            accepted_window / attempted_window if attempted_window > 0 else 0.0
        )
        record_segment[record_count] = schedule_steps.size - 1
        record_accepted_total[record_count] = accepted_total
        record_count += 1

    return (
        occupations,
        best_occupations,
        energy,
        best_energy,
        best_step,
        global_step,
        accepted_total,
        swap_history,
        record_step[:record_count],
        record_temperature[:record_count],
        record_energy[:record_count],
        record_acceptance[:record_count],
        record_segment[:record_count],
        record_accepted_total[:record_count],
    )


def select_active_sites(
    mapped: MappedStructure,
    model: CEModel,
    include_vacancies: bool = False,
    active_layers: Any = "all",
    frozen_species: Iterable[str] = (),
) -> np.ndarray:
    occupations = model.feature_spec.encode_symbols(mapped.symbols)
    mask = np.ones(len(occupations), dtype=bool)
    if not include_vacancies:
        mask &= occupations != model.feature_spec.vacancy_code
    frozen_codes = {
        model.feature_spec.species_to_code[s]
        for s in frozen_species
        if s in model.feature_spec.species_to_code
    }
    for code in frozen_codes:
        mask &= occupations != code

    if active_layers not in (None, "all"):
        if not mapped.kind.startswith("slab"):
            raise ValueError("active_layers is only valid for slab structures")
        if isinstance(active_layers, str):
            raise ValueError("active_layers must be 'all' or a list of integer layer indices")
        occupied_layers = sorted(set(int(x) for x in mapped.layer_index[mapped.metal_mask]))
        resolved: set[int] = set()
        for layer in active_layers:
            layer = int(layer)
            if layer < 0:
                layer = occupied_layers[layer]
            resolved.add(layer)
        mask &= np.isin(mapped.layer_index, sorted(resolved))

    active = np.flatnonzero(mask).astype(np.int32)
    if len(active) < 2:
        raise ValueError("Fewer than two active sites remain after applying move constraints")
    if len(np.unique(occupations[active])) < 2:
        raise ValueError("Active sites contain only one species, so canonical swaps are impossible")
    return active


def _topology_geometry_ids(topology: Topology, model: CEModel) -> np.ndarray:
    if not topology.n_triplets:
        return np.empty(0, dtype=np.int16)
    try:
        return np.fromiter(
            (model.feature_spec.geometry_to_id[g] for g in topology.triplet_geometry),
            dtype=np.int16,
            count=topology.n_triplets,
        )
    except KeyError as exc:
        raise ValueError(f"Annealing topology contains unseen geometry {exc.args[0]}") from exc


def run_annealing(
    model: CEModel,
    mapped: MappedStructure,
    schedule_data: Any,
    *,
    topology: Topology | None = None,
    include_vacancies: bool = False,
    active_layers: Any = "all",
    frozen_species: Iterable[str] = (),
    record_every_sweeps: float = 1.0,
    random_seed: int = 42,
    boltzmann_constant: float = BOLTZMANN_EV_PER_K,
) -> MCResult:
    topology = topology or build_topology(mapped, model.cluster_spec)
    occupations = model.feature_spec.encode_symbols(mapped.symbols)
    active = select_active_sites(
        mapped,
        model,
        include_vacancies=include_vacancies,
        active_layers=active_layers,
        frozen_species=frozen_species,
    )
    schedule = compile_schedule(schedule_data, len(active))
    record_interval = max(1, int(round(float(record_every_sweeps) * len(active))))
    tables = model.coefficient_tables
    geometry_ids = _topology_geometry_ids(topology, model)

    result = _run_mc_kernel(
        occupations.astype(np.int16),
        active,
        int(model.feature_spec.vacancy_code),
        float(tables.zero),
        np.asarray(tables.point, dtype=float),
        topology.pair_i,
        topology.pair_j,
        topology.pair_shell,
        np.asarray(tables.pair, dtype=float),
        topology.triplet_i,
        topology.triplet_j,
        topology.triplet_k,
        geometry_ids,
        np.asarray(tables.triplet, dtype=float),
        topology.site_pair_ptr,
        topology.site_pair_indices,
        topology.site_triplet_ptr,
        topology.site_triplet_indices,
        schedule.profiles,
        schedule.start_temperatures,
        schedule.stop_temperatures,
        schedule.steps,
        record_interval,
        int(random_seed),
        float(boltzmann_constant),
    )
    (
        final_occ,
        best_occ,
        final_energy,
        best_energy,
        best_step,
        attempted,
        accepted,
        swap_history,
        rec_step,
        rec_temperature,
        rec_energy,
        rec_acceptance,
        rec_segment,
        rec_accepted_total,
    ) = result

    n_metal = int(np.count_nonzero(occupations != model.feature_spec.vacancy_code))
    denominator = model.feature_spec.denominator(
        model.target_normalization, n_metal, len(occupations)
    )
    labels = [schedule.labels[int(i)] for i in rec_segment]
    trace = pd.DataFrame(
        {
            "step": rec_step,
            "sweep": rec_step / len(active),
            "segment_index": rec_segment,
            "segment": labels,
            "temperature_K": rec_temperature,
            "total_energy": rec_energy,
            "predicted_target": rec_energy / denominator,
            "acceptance_rate_since_last_record": rec_acceptance,
            "accepted_moves_total": rec_accepted_total,
        }
    )
    return MCResult(
        initial_occupations=np.asarray(occupations, dtype=np.int16).copy(),
        final_occupations=np.asarray(final_occ, dtype=np.int16),
        best_occupations=np.asarray(best_occ, dtype=np.int16),
        swap_history=np.asarray(swap_history, dtype=np.int32),
        trace=trace,
        final_total_energy=float(final_energy),
        best_total_energy=float(best_energy),
        best_step=int(best_step),
        attempted_moves=int(attempted),
        accepted_moves=int(accepted),
    )


def occupations_to_atoms(
    mapped: MappedStructure,
    model: CEModel,
    occupations: np.ndarray,
) -> Atoms:
    atoms = mapped.atoms.copy()
    atoms.set_chemical_symbols([model.feature_spec.species[int(code)] for code in occupations])
    atoms.set_array("ce_layer", mapped.layer_index.astype(np.int32))
    atoms.set_array("ce_site_index", np.arange(len(atoms), dtype=np.int32))
    atoms.info["ce_kind"] = mapped.kind
    atoms.info["ce_nearest_neighbor"] = float(mapped.nearest_neighbor)
    return atoms


def replay_swap_history(
    initial_occupations: np.ndarray,
    swap_history: np.ndarray,
    step: int | None = None,
) -> np.ndarray:
    """Reconstruct occupations after ``step`` completed MC steps.

    ``step=0`` returns the initial state. ``step=-1`` or ``None`` returns the
    final state. Swap-history rows use zero-based parent-lattice site indices;
    ``[-1, -1]`` is the compact representation of JSON ``null``.
    """
    occupations = np.asarray(initial_occupations).copy()
    history = np.asarray(swap_history)
    if occupations.ndim != 1:
        raise ValueError("initial_occupations must be a one-dimensional array")
    if history.ndim != 2 or history.shape[1] != 2:
        raise ValueError("swap_history must have shape (n_steps, 2)")

    total_steps = int(history.shape[0])
    resolved_step = total_steps if step is None or int(step) == -1 else int(step)
    if resolved_step < 0 or resolved_step > total_steps:
        raise ValueError(f"step must be between 0 and {total_steps}, or -1 for final")

    n_sites = int(occupations.size)
    for history_step in range(resolved_step):
        site_a = int(history[history_step, 0])
        site_b = int(history[history_step, 1])
        if site_a == -1 and site_b == -1:
            continue
        if site_a < 0 or site_b < 0 or site_a >= n_sites or site_b >= n_sites:
            raise ValueError(
                f"Invalid swap [{site_a}, {site_b}] at MC step {history_step + 1}"
            )
        if site_a == site_b:
            raise ValueError(f"Self-swap at MC step {history_step + 1}")
        occupations[site_a], occupations[site_b] = (
            occupations[site_b],
            occupations[site_a],
        )
    return occupations


def write_swap_history_json(path: str | Path, swap_history: np.ndarray) -> Path:
    """Write the exact per-step history as ``[[i,j], null, ...]`` JSON.

    The file is streamed so writing it does not require constructing a second
    large Python list in memory.
    """
    history = np.asarray(swap_history)
    if history.ndim != 2 or history.shape[1] != 2:
        raise ValueError("swap_history must have shape (n_steps, 2)")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write("[")
        for index, pair in enumerate(history):
            if index:
                handle.write(",")
            site_a = int(pair[0])
            site_b = int(pair[1])
            if site_a == -1 and site_b == -1:
                handle.write("null")
            else:
                if site_a < 0 or site_b < 0:
                    raise ValueError(f"Invalid partial-null swap at history row {index}")
                handle.write(f"[{site_a},{site_b}]")
        handle.write("]\n")
    return output


def physical_atoms(full_atoms: Atoms, vacancy_symbol: str, kind: str, vacuum_angstrom: float) -> Atoms:
    symbols = np.asarray(full_atoms.get_chemical_symbols())
    atoms = full_atoms[symbols != vacancy_symbol]
    atoms.pbc = True
    if not kind.startswith("slab"):
        return atoms
    cell = np.asarray(full_atoms.cell.array, dtype=float)
    a, b = cell[0].copy(), cell[1].copy()
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    z = atoms.positions @ normal
    span = float(np.max(z) - np.min(z)) if len(z) else 0.0
    height = span + float(vacuum_angstrom)
    shift = float(vacuum_angstrom) / 2.0 - float(np.min(z))
    atoms.positions += normal * shift
    atoms.set_cell(np.vstack([a, b, normal * height]), scale_atoms=False)
    atoms.wrap()
    return atoms


def save_annealing_outputs(
    output_directory: str | Path,
    model: CEModel,
    mapped: MappedStructure,
    result: MCResult,
    *,
    vacuum_angstrom: float = 20.0,
    extra_summary: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    replayed_final = replay_swap_history(result.initial_occupations, result.swap_history)
    if not np.array_equal(replayed_final, result.final_occupations):
        raise ValueError("Swap history does not reproduce the final occupations")

    initial_full = occupations_to_atoms(mapped, model, result.initial_occupations)
    final_full = occupations_to_atoms(mapped, model, result.final_occupations)
    best_full = occupations_to_atoms(mapped, model, result.best_occupations)
    write(output / "initial_full_lattice.extxyz", initial_full)
    write(output / "final_full_lattice.extxyz", final_full)
    write(output / "best_full_lattice.extxyz", best_full)
    write(
        output / "final_physical.cif",
        physical_atoms(final_full, model.feature_spec.vacancy_symbol, mapped.kind, vacuum_angstrom),
    )
    write(
        output / "best_physical.cif",
        physical_atoms(best_full, model.feature_spec.vacancy_symbol, mapped.kind, vacuum_angstrom),
    )
    result.trace.to_csv(output / "anneal_trace.csv", index=False)
    write_swap_history_json(output / "swap_history.json", result.swap_history)
    positions = np.asarray(initial_full.positions, dtype=float)
    pd.DataFrame(
        {
            "site_index": np.arange(len(initial_full), dtype=np.int64),
            "initial_symbol": initial_full.get_chemical_symbols(),
            "layer_index": mapped.layer_index.astype(np.int64),
            "x_A": positions[:, 0],
            "y_A": positions[:, 1],
            "z_A": positions[:, 2],
        }
    ).to_csv(output / "site_index.csv", index=False)
    np.savez_compressed(
        output / "anneal_checkpoint.npz",
        initial_occupations=result.initial_occupations,
        final_occupations=result.final_occupations,
        best_occupations=result.best_occupations,
        swap_history=result.swap_history,
        best_step=np.asarray(result.best_step, dtype=np.int64),
        swap_index_base=np.asarray(0, dtype=np.int8),
    )
    n_metal = mapped.n_metal
    denominator = model.feature_spec.denominator(
        model.target_normalization, n_metal, mapped.n_lattice
    )
    summary = {
        "kind": mapped.kind,
        "n_metal": mapped.n_metal,
        "n_lattice": mapped.n_lattice,
        "attempted_steps": result.attempted_moves,
        "accepted_moves": result.accepted_moves,
        "best_step": result.best_step,
        "overall_acceptance_rate": (
            result.accepted_moves / result.attempted_moves if result.attempted_moves else 0.0
        ),
        "final_total_energy": result.final_total_energy,
        "best_total_energy": result.best_total_energy,
        "final_predicted_target": result.final_total_energy / denominator,
        "best_predicted_target": result.best_total_energy / denominator,
        "target_normalization": model.target_normalization,
        "energy_unit": model.energy_unit,
        "vacancy_symbol": model.feature_spec.vacancy_symbol,
        "swap_history_file": "swap_history.json",
        "swap_history_entries": int(result.swap_history.shape[0]),
        "swap_history_index_base": 0,
        "swap_history_null_meaning": "No swap was performed at this MC step",
        "swap_history_reference_structure": "initial_full_lattice.extxyz",
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(output / "anneal_summary.json", summary)
    return output
