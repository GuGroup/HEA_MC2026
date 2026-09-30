"""Indicator-basis cluster functions for the vacancy-aware fcc CE."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from typing import Any, Iterable

import numpy as np

from .topology import ClusterSpec, Topology, canonical_triplet_decoration


@dataclass
class CoefficientTables:
    zero: float
    point: np.ndarray
    pair: np.ndarray
    triplet: np.ndarray


class FeatureSpec:
    """Defines the complete decorated-cluster vocabulary and lookup tables."""

    def __init__(
        self,
        species: Iterable[str],
        vacancy_symbol: str,
        cluster_spec: ClusterSpec,
        geometries: Iterable[tuple[int, int, int]] = (),
    ) -> None:
        species_list = [str(s) for s in species]
        if vacancy_symbol not in species_list:
            species_list.append(vacancy_symbol)
        # Keep deterministic metal ordering and put the vacancy last.
        metals = sorted(s for s in species_list if s != vacancy_symbol)
        if not metals:
            raise ValueError("At least one non-vacancy species is required")
        self.species = metals + [vacancy_symbol]
        self.vacancy_symbol = vacancy_symbol
        self.vacancy_code = len(self.species) - 1
        self.species_to_code = {s: i for i, s in enumerate(self.species)}
        self.cluster_spec = cluster_spec
        self.geometries = sorted(set(tuple(int(x) for x in g) for g in geometries))
        self.geometry_to_id = {g: i for i, g in enumerate(self.geometries)}

        q = len(self.species)
        names: list[str] = ["zero:n_metal"]

        self.zero_col = 0
        self.point_col = np.full(q, -1, dtype=np.int32)
        for code, symbol in enumerate(self.species):
            if code == self.vacancy_code:
                continue
            self.point_col[code] = len(names)
            names.append(f"point:{symbol}")

        self.pair_col = np.full(
            (self.cluster_spec.pair_max_shell + 1, q, q), -1, dtype=np.int32
        )
        for shell in range(1, self.cluster_spec.pair_max_shell + 1):
            for left, right in combinations_with_replacement(range(q), 2):
                if left == self.vacancy_code and right == self.vacancy_code:
                    continue
                col = len(names)
                self.pair_col[shell, left, right] = col
                self.pair_col[shell, right, left] = col
                names.append(
                    f"pair:shell={shell}:{self.species[left]}-{self.species[right]}"
                )

        self.triplet_col = np.full(
            (len(self.geometries), q, q, q), -1, dtype=np.int32
        )
        for geometry_id, geometry in enumerate(self.geometries):
            decorations = {
                canonical_triplet_decoration(tuple(codes), geometry)
                for codes in product(range(q), repeat=3)
                if not all(code == self.vacancy_code for code in codes)
            }
            decoration_to_col: dict[tuple[int, int, int], int] = {}
            for decoration in sorted(decorations):
                col = len(names)
                decoration_to_col[decoration] = col
                symbols = "-".join(self.species[c] for c in decoration)
                names.append(
                    f"triplet:geometry={geometry[0]}-{geometry[1]}-{geometry[2]}:{symbols}"
                )
            for codes in product(range(q), repeat=3):
                if all(code == self.vacancy_code for code in codes):
                    continue
                decoration = canonical_triplet_decoration(tuple(codes), geometry)
                self.triplet_col[(geometry_id, *codes)] = decoration_to_col[decoration]

        self.feature_names = names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_species(self) -> int:
        return len(self.species)

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "vacancy_symbol": self.vacancy_symbol,
            "cluster_spec": self.cluster_spec.to_dict(),
            "geometries": [list(g) for g in self.geometries],
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        obj = cls(
            species=data["species"],
            vacancy_symbol=data["vacancy_symbol"],
            cluster_spec=ClusterSpec.from_dict(data["cluster_spec"]),
            geometries=[tuple(g) for g in data.get("geometries", [])],
        )
        expected = data.get("feature_names")
        if expected is not None and list(expected) != obj.feature_names:
            raise ValueError("Saved feature vocabulary is inconsistent with reconstructed tables")
        return obj

    def encode_symbols(self, symbols: Iterable[str]) -> np.ndarray:
        codes = []
        unknown = set()
        for symbol in symbols:
            code = self.species_to_code.get(str(symbol))
            if code is None:
                unknown.add(str(symbol))
                codes.append(-1)
            else:
                codes.append(code)
        if unknown:
            raise ValueError(
                f"Structure contains species not present in the model: {sorted(unknown)}; "
                f"allowed={self.species}"
            )
        return np.asarray(codes, dtype=np.int16)

    def raw_counts(self, symbols: Iterable[str], topology: Topology) -> tuple[np.ndarray, np.ndarray]:
        occupations = self.encode_symbols(symbols)
        if len(occupations) != topology.n_sites:
            raise ValueError("Occupation vector and topology have different sizes")
        counts = np.zeros(self.n_features, dtype=float)
        metal_mask = occupations != self.vacancy_code
        n_metal = int(np.count_nonzero(metal_mask))
        if n_metal == 0:
            raise ValueError("A structure with no metal atoms cannot be featurized")
        counts[self.zero_col] = n_metal

        point_counts = np.bincount(occupations, minlength=self.n_species)
        for code, col in enumerate(self.point_col):
            if col >= 0:
                counts[col] = point_counts[code]

        if topology.n_pairs:
            cols = self.pair_col[
                topology.pair_shell,
                occupations[topology.pair_i],
                occupations[topology.pair_j],
            ]
            valid = cols >= 0
            if np.any(valid):
                np.add.at(counts, cols[valid], 1.0)

        if topology.n_triplets:
            try:
                geom_ids = np.fromiter(
                    (self.geometry_to_id[g] for g in topology.triplet_geometry),
                    dtype=np.int32,
                    count=topology.n_triplets,
                )
            except KeyError as exc:
                raise ValueError(f"Topology contains an unseen triplet geometry: {exc.args[0]}") from exc
            cols = self.triplet_col[
                geom_ids,
                occupations[topology.triplet_i],
                occupations[topology.triplet_j],
                occupations[topology.triplet_k],
            ]
            valid = cols >= 0
            if np.any(valid):
                np.add.at(counts, cols[valid], 1.0)

        return counts, occupations

    @staticmethod
    def denominator(normalization: str, n_metal: int, n_lattice: int) -> float:
        mode = normalization.lower()
        if mode == "per_atom":
            return float(n_metal)
        if mode == "per_lattice_site":
            return float(n_lattice)
        if mode == "total":
            return 1.0
        raise ValueError("target_normalization must be per_atom, per_lattice_site, or total")

    def vector(
        self,
        symbols: Iterable[str],
        topology: Topology,
        normalization: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        counts, occupations = self.raw_counts(symbols, topology)
        n_metal = int(np.count_nonzero(occupations != self.vacancy_code))
        denom = self.denominator(normalization, n_metal, len(occupations))
        return counts / denom, occupations, denom

    def coefficient_tables(self, coefficients: np.ndarray) -> CoefficientTables:
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.n_features,):
            raise ValueError(
                f"Expected {self.n_features} coefficients, got shape {coefficients.shape}"
            )
        q = self.n_species
        point = np.zeros(q, dtype=float)
        for code, col in enumerate(self.point_col):
            if col >= 0:
                point[code] = coefficients[col]

        pair = np.zeros_like(self.pair_col, dtype=float)
        valid = self.pair_col >= 0
        pair[valid] = coefficients[self.pair_col[valid]]

        triplet = np.zeros_like(self.triplet_col, dtype=float)
        valid = self.triplet_col >= 0
        triplet[valid] = coefficients[self.triplet_col[valid]]
        return CoefficientTables(
            zero=float(coefficients[self.zero_col]),
            point=point,
            pair=pair,
            triplet=triplet,
        )
