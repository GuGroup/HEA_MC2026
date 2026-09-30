"""Automatic reconstruction of bulk and slab structures on an fcc parent lattice.

The mapper accepts ordinary atom-only CIF files.  For slabs it detects the vacuum
axis, unwraps the occupied layers, identifies fcc(111), fcc(100), or fcc(110), snaps each
layer to a repeating stacking template, and appends explicit vacancy (``X``)
layers.  The resulting fully periodic lattice is suitable for a local cluster
expansion and Monte Carlo sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import math

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from scipy.optimize import linear_sum_assignment


VACANCY_SYMBOL = "X"


@dataclass(slots=True)
class MapperConfig:
    vacancy_symbol: str = VACANCY_SYMBOL
    slab_vacuum_layers: int = 6
    layer_tolerance_angstrom: float = 0.25
    xy_match_tolerance_angstrom: float = 0.20
    slab_gap_factor: float = 1.8
    orientation_ratio_tolerance: float = 0.10
    force_kind: str = "auto"  # auto, bulk, slab111, slab100, slab110

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MapperConfig":
        if not data:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown lattice mapper settings: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MappedStructure:
    atoms: Atoms
    kind: str
    nearest_neighbor: float
    layer_index: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vacancy_symbol(self) -> str:
        return str(self.metadata.get("vacancy_symbol", VACANCY_SYMBOL))

    @property
    def symbols(self) -> list[str]:
        return self.atoms.get_chemical_symbols()

    @property
    def n_lattice(self) -> int:
        return len(self.atoms)

    @property
    def vacancy_mask(self) -> np.ndarray:
        return np.asarray(self.symbols) == self.vacancy_symbol

    @property
    def metal_mask(self) -> np.ndarray:
        return ~self.vacancy_mask

    @property
    def n_metal(self) -> int:
        return int(np.count_nonzero(self.metal_mask))

    def repeat(self, repeat: Iterable[int]) -> "MappedStructure":
        rep = tuple(int(x) for x in repeat)
        if len(rep) != 3 or min(rep) < 1:
            raise ValueError("repeat must contain three positive integers")
        atoms = self.atoms.repeat(rep)
        # ASE propagates per-atom arrays through repeat, but layer_index is kept
        # separately too, so reconstruct it explicitly for portability.
        layer_index = np.tile(self.layer_index, int(np.prod(rep)))
        metadata = dict(self.metadata)
        metadata["repeat"] = list(rep)
        return MappedStructure(
            atoms=atoms,
            kind=self.kind,
            nearest_neighbor=self.nearest_neighbor,
            layer_index=layer_index,
            metadata=metadata,
        )


def read_structure(path: str | Path) -> Atoms:
    atoms = read(str(path))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected one structure in {path!s}")
    if len(atoms) == 0:
        raise ValueError(f"Structure {path!s} contains no atoms")
    atoms.pbc = True
    return atoms


def _estimate_nearest_neighbor(atoms: Atoms, vacancy_symbol: str = VACANCY_SYMBOL) -> float:
    """Estimate the shortest metal-metal distance with O(min(N,64)*N) work."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    # In an explicit-vacancy parent lattice, X positions are genuine lattice
    # sites and give the most reliable spacing even after vacancy diffusion.
    sites = np.arange(len(atoms), dtype=int) if np.any(symbols == vacancy_symbol) else np.flatnonzero(symbols != vacancy_symbol)
    if len(sites) < 2:
        raise ValueError("At least two lattice sites are required")
    sample = sites[: min(64, len(sites))]
    nearest: list[float] = []
    positions = atoms.positions
    for i in sample:
        others = sites[sites != i]
        vectors = positions[others] - positions[i]
        _, distances = find_mic(vectors, atoms.cell, pbc=True)
        positive = distances[distances > 1e-6]
        if len(positive):
            nearest.append(float(np.min(positive)))
    if not nearest:
        raise ValueError("Could not estimate nearest-neighbor distance")
    # Median nearest-neighbor distance is much less sensitive than the absolute
    # minimum to a locally relaxed or slightly noisy input geometry.
    return float(np.median(nearest))


def _axis_geometry(atoms: Atoms, axis: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Return normal, cell height, projected coordinates, unwrapped coords and gap."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    inplane = [i for i in range(3) if i != axis]
    normal = np.cross(cell[inplane[0]], cell[inplane[1]])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        raise ValueError("Degenerate simulation cell")
    normal /= norm
    if float(np.dot(cell[axis], normal)) < 0:
        normal *= -1.0
    height = float(abs(np.dot(cell[axis], normal)))
    if height < 1e-10:
        raise ValueError("Cell vector has no component normal to the other two")
    projected = np.mod(atoms.positions @ normal, height)
    order = np.argsort(projected)
    sorted_z = projected[order]
    gaps = np.diff(np.r_[sorted_z, sorted_z[0] + height])
    gap_index = int(np.argmax(gaps))
    largest_gap = float(gaps[gap_index])
    start = float(sorted_z[(gap_index + 1) % len(sorted_z)])
    unwrapped = np.mod(projected - start, height)
    return normal, height, projected, unwrapped, largest_gap


def _find_slab_axis(atoms: Atoms, nn: float, cfg: MapperConfig) -> tuple[int | None, dict[int, float]]:
    gaps: dict[int, float] = {}
    for axis in range(3):
        try:
            _, height, _, unwrapped, gap = _axis_geometry(atoms, axis)
        except ValueError:
            continue
        occupied_span = float(np.max(unwrapped) - np.min(unwrapped)) if len(unwrapped) else 0.0
        # Both a genuinely large circular gap and a cell height larger than the
        # occupied span are required.  This rejects ordinary elongated bulk cells.
        score = gap / nn
        if height > occupied_span + cfg.slab_gap_factor * nn:
            gaps[axis] = score
    if not gaps:
        return None, {}
    axis = max(gaps, key=gaps.get)
    if gaps[axis] < cfg.slab_gap_factor:
        return None, gaps
    return axis, gaps


def _cluster_layers(z: np.ndarray, tolerance: float) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.argsort(z)
    groups: list[list[int]] = []
    centers: list[float] = []
    for index in order:
        value = float(z[index])
        if not groups or abs(value - centers[-1]) > tolerance:
            groups.append([int(index)])
            centers.append(value)
        else:
            groups[-1].append(int(index))
            centers[-1] = float(np.mean(z[groups[-1]]))
    arrays = [np.asarray(g, dtype=int) for g in groups]
    return arrays, np.asarray(centers, dtype=float)


def _periodic_xy_distances(xy_a: np.ndarray, xy_b: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise minimum in-plane distances between fractional 2D coordinates."""
    delta = xy_a[:, None, :] - xy_b[None, :, :]
    best = np.full(delta.shape[:2], np.inf, dtype=float)
    for tx in (-1, 0, 1):
        for ty in (-1, 0, 1):
            shifted = delta + np.array([tx, ty], dtype=float)
            cart = shifted[..., 0, None] * a + shifted[..., 1, None] * b
            best = np.minimum(best, np.linalg.norm(cart, axis=-1))
    return best


def _plane_fractional_coordinates(positions: np.ndarray, a: np.ndarray, b: np.ndarray, normal: np.ndarray) -> np.ndarray:
    projected = positions - np.outer(positions @ normal, normal)
    basis = np.column_stack([a, b])  # (3, 2)
    coeff, *_ = np.linalg.lstsq(basis, projected.T, rcond=None)
    return np.mod(coeff.T, 1.0)


def _sort_xy(xy: np.ndarray) -> np.ndarray:
    rounded = np.round(np.mod(xy, 1.0), 10)
    return np.lexsort((rounded[:, 1], rounded[:, 0]))


def _classify_surface(
    inplane_a: np.ndarray,
    inplane_b: np.ndarray,
    d_layer: float,
    nn: float,
    cfg: MapperConfig,
) -> str:
    ratio = d_layer / nn
    err111 = abs(ratio - math.sqrt(2.0 / 3.0))
    err100 = abs(ratio - 1.0 / math.sqrt(2.0))
    err110 = abs(ratio - 0.5)
    cosine = float(np.dot(inplane_a, inplane_b) / (np.linalg.norm(inplane_a) * np.linalg.norm(inplane_b)))
    angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

    candidates: list[tuple[float, str]] = [(err111, "slab111"), (err100, "slab100"), (err110, "slab110")]
    best_err, best_kind = min(candidates, key=lambda item: item[0])

    # Use layer spacing as the primary identifier.  fcc(111), fcc(100), and
    # fcc(110) have d_hkl/d_nn values sqrt(2/3), 1/sqrt(2), and 1/2,
    # respectively.  The in-plane angle is used only as a fallback for heavily
    # rounded or unusual cells.
    if best_err <= cfg.orientation_ratio_tolerance:
        return best_kind
    if min(abs(angle - 60.0), abs(angle - 120.0)) < 12.0:
        return "slab111"
    if abs(angle - 90.0) < 12.0:
        # Rectangular cells could be either fcc(100) or fcc(110).  If the layer
        # spacing is closer to d_nn/2 than d_nn/sqrt(2), classify as (110).
        return "slab110" if err110 < err100 else "slab100"
    raise ValueError(
        f"Could not identify fcc surface: d_layer/d_nn={ratio:.4f}, in-plane angle={angle:.2f}°"
    )


def _canonical_sort(atoms: Atoms, layer_index: np.ndarray | None = None) -> tuple[Atoms, np.ndarray]:
    scaled = np.mod(atoms.get_scaled_positions(wrap=True), 1.0)
    rounded = np.round(scaled, 10)
    if layer_index is None:
        layer_index = np.full(len(atoms), -1, dtype=np.int32)
        order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    else:
        layer_index = np.asarray(layer_index, dtype=np.int32)
        order = np.lexsort((rounded[:, 1], rounded[:, 0], layer_index))
    sorted_atoms = atoms[order]
    sorted_atoms.cell = atoms.cell
    sorted_atoms.pbc = True
    sorted_layers = layer_index[order]
    sorted_atoms.set_array("ce_layer", sorted_layers.astype(np.int32))
    return sorted_atoms, sorted_layers


def _map_bulk(atoms: Atoms, nn: float, cfg: MapperConfig) -> MappedStructure:
    mapped, layers = _canonical_sort(atoms.copy())
    metadata = {
        "vacancy_symbol": cfg.vacancy_symbol,
        "source_n_atoms": len(atoms),
        "source_cell": np.asarray(atoms.cell.array).tolist(),
        "mapping": "bulk_passthrough_canonical_sort",
    }
    mapped.info["ce_kind"] = "bulk"
    mapped.info["ce_nearest_neighbor"] = float(nn)
    return MappedStructure(mapped, "bulk", nn, layers, metadata)


def _map_slab(atoms: Atoms, axis: int, nn: float, cfg: MapperConfig, forced_kind: str | None = None) -> MappedStructure:
    cell = np.asarray(atoms.cell.array, dtype=float)
    inplane_axes = [i for i in range(3) if i != axis]
    a = cell[inplane_axes[0]].copy()
    b = cell[inplane_axes[1]].copy()
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    if np.dot(cell[axis], normal) < 0:
        b *= -1.0
        normal = np.cross(a, b)
        normal /= np.linalg.norm(normal)

    _, original_height, _, unwrapped, largest_gap = _axis_geometry(atoms, axis)
    groups, centers = _cluster_layers(unwrapped, cfg.layer_tolerance_angstrom)
    if len(groups) < 2:
        raise ValueError("A slab requires at least two occupied atomic layers")
    counts = np.asarray([len(g) for g in groups], dtype=int)
    expected = int(np.median(counts))
    if np.any(counts != expected):
        raise ValueError(
            "The detected slab layers do not contain the same number of fcc sites: "
            f"{counts.tolist()}. Reconstructions/adsorbates need a custom mapper."
        )
    spacings = np.diff(centers)
    d_layer = float(np.median(spacings))
    if np.max(np.abs(spacings - d_layer)) > max(cfg.layer_tolerance_angstrom, 0.08 * d_layer):
        raise ValueError(f"Non-uniform slab layer spacings detected: {spacings.tolist()}")

    kind = forced_kind or _classify_surface(a, b, d_layer, nn, cfg)
    preferred_period = 3 if kind == "slab111" else 2
    n_occupied = len(groups)

    xy_all = _plane_fractional_coordinates(atoms.positions, a, b, normal)
    source_symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)

    def try_stacking_period(period_candidate: int) -> tuple[list[np.ndarray], list[list[str]], float]:
        if n_occupied < period_candidate:
            raise ValueError(f"At least {period_candidate} occupied layers are required")
        refs: list[np.ndarray] = []
        for phase in range(period_candidate):
            xy = np.mod(xy_all[groups[phase]], 1.0)
            xy = xy[_sort_xy(xy)]
            refs.append(xy)
        layer_symbols: list[list[str]] = []
        worst = 0.0
        for layer, indices in enumerate(groups):
            phase = layer % period_candidate
            xy_source = np.mod(xy_all[indices], 1.0)
            ref = refs[phase]
            distances = _periodic_xy_distances(xy_source, ref, a, b)
            rows, cols = linear_sum_assignment(distances)
            matched = distances[rows, cols]
            current_worst = float(np.max(matched)) if len(matched) else 0.0
            worst = max(worst, current_worst)
            if current_worst > cfg.xy_match_tolerance_angstrom:
                raise ValueError(
                    f"Layer {layer} cannot be snapped to the {kind} template with stacking period "
                    f"{period_candidate}; maximum in-plane displacement={current_worst:.3f} Å"
                )
            symbols = [cfg.vacancy_symbol] * expected
            for row, col in zip(rows, cols):
                symbols[int(col)] = str(source_symbols[indices[int(row)]])
            layer_symbols.append(symbols)
        return refs, layer_symbols, worst

    candidate_periods = [preferred_period]
    if forced_kind is not None:
        # When a caller explicitly labels a surface as slab110, accept unusual
        # exported cells whose layer registry repeats with a different period.
        # This keeps the family label requested by the user while still using the
        # actual periodic parent lattice encoded in the CIF.
        candidate_periods.extend([1, 2, 3, 4, 6])
    candidate_periods = list(dict.fromkeys(p for p in candidate_periods if p <= n_occupied))
    errors: list[str] = []
    for period in candidate_periods:
        try:
            reference_xy, occupied_symbols, max_match_error = try_stacking_period(period)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        detail = "; ".join(errors[-3:])
        raise ValueError(f"Could not snap occupied layers to a {kind} stacking template. {detail}")

    n_vacancy = max(1, int(cfg.slab_vacuum_layers))
    n_total = n_occupied + n_vacancy
    # A cell vector normal to (111)/(100)/(110) closes the stacking only after
    # a multiple of the surface stacking period.  The fcc(111) ABC period is 3;
    # fcc(100) and fcc(110) use a two-layer template in this mapper.
    if n_total % period:
        n_total += period - (n_total % period)
    n_vacancy = n_total - n_occupied

    all_positions: list[np.ndarray] = []
    all_symbols: list[str] = []
    all_layers: list[int] = []
    for layer in range(n_total):
        phase = layer % period
        xy = reference_xy[phase]
        zvector = normal * (layer * d_layer)
        positions = xy[:, 0, None] * a + xy[:, 1, None] * b + zvector
        all_positions.extend(positions)
        if layer < n_occupied:
            all_symbols.extend(occupied_symbols[layer])
        else:
            all_symbols.extend([cfg.vacancy_symbol] * expected)
        all_layers.extend([layer] * expected)

    mapped_cell = np.vstack([a, b, normal * (n_total * d_layer)])
    mapped = Atoms(all_symbols, positions=np.asarray(all_positions), cell=mapped_cell, pbc=True)
    mapped, layer_index = _canonical_sort(mapped, np.asarray(all_layers, dtype=np.int32))
    mapped.info["ce_kind"] = kind
    mapped.info["ce_nearest_neighbor"] = float(nn)
    mapped.info["ce_occupied_layers"] = int(n_occupied)
    mapped.info["ce_total_layers"] = int(n_total)

    metadata = {
        "vacancy_symbol": cfg.vacancy_symbol,
        "source_n_atoms": len(atoms),
        "source_cell": cell.tolist(),
        "source_vacuum_axis": int(axis),
        "source_cell_height": original_height,
        "source_largest_gap": largest_gap,
        "surface_orientation": kind.removeprefix("slab"),
        "stacking_period": period,
        "sites_per_layer": expected,
        "occupied_layers": n_occupied,
        "vacancy_layers": n_vacancy,
        "total_layers": n_total,
        "layer_spacing": d_layer,
        "max_xy_snap_displacement": max_match_error,
        "mapping": "fcc_layer_template_with_explicit_vacancies",
    }
    return MappedStructure(mapped, kind, nn, layer_index, metadata)


def _map_existing_full_lattice(atoms: Atoms, nn: float, cfg: MapperConfig) -> MappedStructure:
    """Accept a previously written full-lattice structure containing X sites."""
    info_kind = str(atoms.info.get("ce_kind", ""))
    if info_kind in {"bulk", "slab111", "slab100", "slab110"}:
        kind = info_kind
    else:
        lengths = np.asarray(atoms.cell.lengths())
        axis = int(np.argmax(lengths))
        cell = np.asarray(atoms.cell.array)
        inplane = [i for i in range(3) if i != axis]
        normal = np.cross(cell[inplane[0]], cell[inplane[1]])
        normal /= np.linalg.norm(normal)
        z = np.mod(atoms.positions @ normal, abs(np.dot(cell[axis], normal)))
        groups, centers = _cluster_layers(z, cfg.layer_tolerance_angstrom)
        if len(groups) < 2:
            raise ValueError("Could not infer layers from the explicit-vacancy structure")
        d_layer = float(np.median(np.diff(np.sort(centers))))
        kind = _classify_surface(cell[inplane[0]], cell[inplane[1]], d_layer, nn, cfg)
    if "ce_layer" in atoms.arrays:
        layers = np.asarray(atoms.arrays["ce_layer"], dtype=np.int32)
    elif kind.startswith("slab"):
        axis = int(np.argmax(atoms.cell.lengths()))
        normal, _, _, unwrapped, _ = _axis_geometry(atoms, axis)
        groups, _ = _cluster_layers(unwrapped, cfg.layer_tolerance_angstrom)
        layers = np.empty(len(atoms), dtype=np.int32)
        for k, indices in enumerate(groups):
            layers[indices] = k
    else:
        layers = np.full(len(atoms), -1, dtype=np.int32)
    mapped, layers = _canonical_sort(atoms.copy(), layers)
    metadata = {
        "vacancy_symbol": cfg.vacancy_symbol,
        "mapping": "existing_explicit_vacancy_lattice",
    }
    return MappedStructure(mapped, kind, nn, layers, metadata)


def map_atoms(atoms: Atoms, cfg: MapperConfig | None = None) -> MappedStructure:
    cfg = cfg or MapperConfig()
    atoms = atoms.copy()
    atoms.pbc = True
    stored_nn = atoms.info.get("ce_nearest_neighbor")
    nn = float(stored_nn) if stored_nn is not None else _estimate_nearest_neighbor(atoms, cfg.vacancy_symbol)
    symbols = atoms.get_chemical_symbols()

    if cfg.vacancy_symbol in symbols:
        return _map_existing_full_lattice(atoms, nn, cfg)

    forced = cfg.force_kind.lower()
    if forced not in {"auto", "bulk", "slab111", "slab100", "slab110"}:
        raise ValueError("force_kind must be one of auto, bulk, slab111, slab100, slab110")
    if forced == "bulk":
        return _map_bulk(atoms, nn, cfg)

    axis, _ = _find_slab_axis(atoms, nn, cfg)
    if forced.startswith("slab"):
        if axis is None:
            # For explicitly forced slabs, use the cell axis with the largest
            # perpendicular height even if the vacuum is small.
            heights = []
            for candidate in range(3):
                try:
                    _, height, *_ = _axis_geometry(atoms, candidate)
                except ValueError:
                    height = -math.inf
                heights.append(height)
            axis = int(np.argmax(heights))
        return _map_slab(atoms, axis, nn, cfg, forced_kind=forced)

    if axis is None:
        return _map_bulk(atoms, nn, cfg)
    return _map_slab(atoms, axis, nn, cfg)


def map_structure(path: str | Path, cfg: MapperConfig | None = None) -> MappedStructure:
    return map_atoms(read_structure(path), cfg)
