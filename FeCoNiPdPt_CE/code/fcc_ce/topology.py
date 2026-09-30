"""Periodic fcc pair/triplet topology construction.

Clusters include periodic images, not merely minimum-image atom pairs.  This is
important when a training supercell is comparable to the requested cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, permutations
from typing import Iterable

import numpy as np
from ase.neighborlist import neighbor_list

from .lattice import MappedStructure


_PERMUTATIONS = tuple(permutations(range(3)))


@dataclass(slots=True)
class ClusterSpec:
    pair_max_shell: int = 4
    triplet_max_shell: int = 2
    include_triplets: bool = True
    shell_tolerance: float = 0.08

    @classmethod
    def from_dict(cls, data: dict | None) -> "ClusterSpec":
        if not data:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown cluster settings: {unknown}")
        obj = cls(**data)
        if obj.pair_max_shell < 1:
            raise ValueError("pair_max_shell must be at least 1")
        if obj.include_triplets and obj.triplet_max_shell < 1:
            raise ValueError("triplet_max_shell must be at least 1")
        return obj

    def to_dict(self) -> dict:
        return {
            "pair_max_shell": int(self.pair_max_shell),
            "triplet_max_shell": int(self.triplet_max_shell),
            "include_triplets": bool(self.include_triplets),
            "shell_tolerance": float(self.shell_tolerance),
        }


@dataclass
class Topology:
    n_sites: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_shell: np.ndarray
    triplet_i: np.ndarray
    triplet_j: np.ndarray
    triplet_k: np.ndarray
    triplet_geometry: list[tuple[int, int, int]]
    site_pair_ptr: np.ndarray
    site_pair_indices: np.ndarray
    site_triplet_ptr: np.ndarray
    site_triplet_indices: np.ndarray
    signature: str

    @property
    def n_pairs(self) -> int:
        return int(len(self.pair_i))

    @property
    def n_triplets(self) -> int:
        return int(len(self.triplet_i))

    @property
    def geometry_keys(self) -> list[tuple[int, int, int]]:
        return sorted(set(self.triplet_geometry))


def topology_signature(mapped: MappedStructure) -> str:
    nn = float(mapped.nearest_neighbor)
    normalized_cell = np.asarray(mapped.atoms.cell.array, dtype=float) / nn
    scaled = np.mod(mapped.atoms.get_scaled_positions(wrap=True), 1.0)
    # The mapper canonically sorts positions, but sort again for a stable hash if
    # a caller constructed MappedStructure manually.
    order = np.lexsort((np.round(scaled[:, 2], 9), np.round(scaled[:, 1], 9), np.round(scaled[:, 0], 9)))
    payload = np.r_[np.round(normalized_cell.ravel(), 8), np.round(scaled[order].ravel(), 8)]
    return sha256(payload.tobytes()).hexdigest()


def _shell_index(distance: float, nn: float, tolerance: float) -> int | None:
    value = (float(distance) / nn) ** 2
    shell = int(round(value))
    if shell < 1 or abs(value - shell) > tolerance:
        return None
    return shell


def _undirected_pair_key(i: int, j: int, shift: Iterable[int]) -> tuple[int, int, int, int, int]:
    s = tuple(int(x) for x in shift)
    forward = (int(i), int(j), s[0], s[1], s[2])
    reverse = (int(j), int(i), -s[0], -s[1], -s[2])
    return min(forward, reverse)


def _canonical_periodic_cluster(sites: list[tuple[int, np.ndarray]]) -> tuple[tuple[int, int, int, int], ...]:
    candidates: list[tuple[tuple[int, int, int, int], ...]] = []
    for anchor in range(len(sites)):
        origin = sites[anchor][1]
        translated = []
        for index, shift in sites:
            ds = np.asarray(shift, dtype=int) - origin
            translated.append((int(index), int(ds[0]), int(ds[1]), int(ds[2])))
        candidates.append(tuple(sorted(translated)))
    return min(candidates)


def _edge_tuple(matrix: np.ndarray, perm: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        int(matrix[perm[0], perm[1]]),
        int(matrix[perm[0], perm[2]]),
        int(matrix[perm[1], perm[2]]),
    )


def canonicalize_triangle_vertices(
    vertices: tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], tuple[int, np.ndarray]],
    edge_matrix: np.ndarray,
) -> tuple[tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], tuple[int, np.ndarray]], tuple[int, int, int]]:
    edge_representations = [(_edge_tuple(edge_matrix, p), p) for p in _PERMUTATIONS]
    geometry = min(item[0] for item in edge_representations)
    candidates = [p for edge, p in edge_representations if edge == geometry]
    # Tie-break equivalent geometric orderings by periodic site labels.  Energy
    # lookup remains invariant because coefficients are expanded over the full
    # automorphism group of the geometry.
    def label_for(p: tuple[int, int, int]) -> tuple:
        return tuple((vertices[x][0], *tuple(int(v) for v in vertices[x][1])) for x in p)

    chosen = min(candidates, key=label_for)
    reordered = tuple(vertices[x] for x in chosen)
    return reordered, geometry


def geometry_automorphisms(geometry: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    matrix = np.zeros((3, 3), dtype=int)
    matrix[0, 1] = matrix[1, 0] = int(geometry[0])
    matrix[0, 2] = matrix[2, 0] = int(geometry[1])
    matrix[1, 2] = matrix[2, 1] = int(geometry[2])
    return tuple(p for p in _PERMUTATIONS if _edge_tuple(matrix, p) == geometry)


def canonical_triplet_decoration(
    species_codes: tuple[int, int, int], geometry: tuple[int, int, int]
) -> tuple[int, int, int]:
    automorphisms = geometry_automorphisms(geometry)
    return min(tuple(int(species_codes[p]) for p in perm) for perm in automorphisms)


def _make_incidence(n_sites: int, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lists: list[list[int]] = [[] for _ in range(n_sites)]
    for cluster_index, row in enumerate(clusters):
        for site in sorted(set(int(x) for x in row)):
            lists[site].append(cluster_index)
    ptr = np.zeros(n_sites + 1, dtype=np.int64)
    for site, values in enumerate(lists):
        ptr[site + 1] = ptr[site] + len(values)
    if ptr[-1]:
        indices = np.concatenate([np.asarray(v, dtype=np.int32) for v in lists])
    else:
        indices = np.empty(0, dtype=np.int32)
    return ptr, indices


def build_topology(mapped: MappedStructure, spec: ClusterSpec) -> Topology:
    atoms = mapped.atoms
    nn = float(mapped.nearest_neighbor)
    max_shell = max(spec.pair_max_shell, spec.triplet_max_shell if spec.include_triplets else 0)
    cutoff = nn * np.sqrt(max_shell) * (1.0 + max(0.02, spec.shell_tolerance))
    i_all, j_all, shifts_all, distances_all = neighbor_list(
        "ijSd", atoms, cutoff, self_interaction=False
    )

    pair_records: dict[tuple[int, int, int, int, int], int] = {}
    for i, j, shift, distance in zip(i_all, j_all, shifts_all, distances_all):
        shell = _shell_index(float(distance), nn, spec.shell_tolerance)
        if shell is None or shell > spec.pair_max_shell:
            continue
        key = _undirected_pair_key(int(i), int(j), shift)
        pair_records[key] = shell

    pair_keys = sorted(pair_records)
    pair_i = np.asarray([key[0] for key in pair_keys], dtype=np.int32)
    pair_j = np.asarray([key[1] for key in pair_keys], dtype=np.int32)
    pair_shell = np.asarray([pair_records[key] for key in pair_keys], dtype=np.int16)

    triplet_rows: list[tuple[int, int, int]] = []
    triplet_geometry: list[tuple[int, int, int]] = []
    if spec.include_triplets:
        # Directed neighbor images grouped by an anchor atom in the home cell.
        neighbors: list[list[tuple[int, np.ndarray, np.ndarray, int]]] = [
            [] for _ in range(len(atoms))
        ]
        cell = np.asarray(atoms.cell.array, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        for i, j, shift, distance in zip(i_all, j_all, shifts_all, distances_all):
            shell = _shell_index(float(distance), nn, spec.shell_tolerance)
            if shell is None or shell > spec.triplet_max_shell:
                continue
            shift = np.asarray(shift, dtype=np.int32)
            vector = positions[int(j)] + shift @ cell - positions[int(i)]
            neighbors[int(i)].append((int(j), shift, vector, shell))

        unique: dict[
            tuple[tuple[int, int, int, int], ...],
            tuple[tuple[tuple[int, np.ndarray], tuple[int, np.ndarray], tuple[int, np.ndarray]], tuple[int, int, int]],
        ] = {}
        for anchor, entries in enumerate(neighbors):
            for left, right in combinations(entries, 2):
                j, shift_j, vector_j, shell_ij = left
                k, shift_k, vector_k, shell_ik = right
                if j == k and np.array_equal(shift_j, shift_k):
                    continue
                shell_jk = _shell_index(float(np.linalg.norm(vector_k - vector_j)), nn, spec.shell_tolerance)
                if shell_jk is None or shell_jk > spec.triplet_max_shell:
                    continue
                sites = [
                    (int(anchor), np.zeros(3, dtype=np.int32)),
                    (int(j), np.asarray(shift_j, dtype=np.int32)),
                    (int(k), np.asarray(shift_k, dtype=np.int32)),
                ]
                cluster_key = _canonical_periodic_cluster(sites)
                if cluster_key in unique:
                    continue
                edge_matrix = np.zeros((3, 3), dtype=np.int16)
                edge_matrix[0, 1] = edge_matrix[1, 0] = int(shell_ij)
                edge_matrix[0, 2] = edge_matrix[2, 0] = int(shell_ik)
                edge_matrix[1, 2] = edge_matrix[2, 1] = int(shell_jk)
                canonical_vertices, geometry = canonicalize_triangle_vertices(
                    (sites[0], sites[1], sites[2]), edge_matrix
                )
                unique[cluster_key] = (canonical_vertices, geometry)

        for key in sorted(unique):
            vertices, geometry = unique[key]
            triplet_rows.append(tuple(int(v[0]) for v in vertices))
            triplet_geometry.append(tuple(int(x) for x in geometry))

    if triplet_rows:
        triplet_array = np.asarray(triplet_rows, dtype=np.int32)
        triplet_i = triplet_array[:, 0]
        triplet_j = triplet_array[:, 1]
        triplet_k = triplet_array[:, 2]
    else:
        triplet_array = np.empty((0, 3), dtype=np.int32)
        triplet_i = np.empty(0, dtype=np.int32)
        triplet_j = np.empty(0, dtype=np.int32)
        triplet_k = np.empty(0, dtype=np.int32)

    pair_clusters = np.column_stack([pair_i, pair_j]) if len(pair_i) else np.empty((0, 2), dtype=np.int32)
    site_pair_ptr, site_pair_indices = _make_incidence(len(atoms), pair_clusters)
    site_triplet_ptr, site_triplet_indices = _make_incidence(len(atoms), triplet_array)

    return Topology(
        n_sites=len(atoms),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_shell=pair_shell,
        triplet_i=triplet_i,
        triplet_j=triplet_j,
        triplet_k=triplet_k,
        triplet_geometry=triplet_geometry,
        site_pair_ptr=site_pair_ptr,
        site_pair_indices=site_pair_indices,
        site_triplet_ptr=site_triplet_ptr,
        site_triplet_indices=site_triplet_indices,
        signature=topology_signature(mapped),
    )
