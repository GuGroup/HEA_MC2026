"""Training, serialization, and prediction for the fcc cluster expansion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .config import write_json
from .data import discover_structures, match_targets, read_targets
from .features import CoefficientTables, FeatureSpec
from .lattice import MapperConfig, MappedStructure, map_structure
from .topology import ClusterSpec, Topology, build_topology, topology_signature


MODEL_FORMAT_VERSION = 1


def _numeric_identifier(identifier: str) -> int | None:
    stem = Path(identifier).stem
    try:
        return int(stem)
    except ValueError:
        match = re.search(r"(\d+)", stem)
        return int(match.group(1)) if match else None


def _force_kind_for_identifier(identifier: str, rules: list[dict[str, Any]]) -> str | None:
    """Return a per-structure mapper force_kind from data.force_kind_by_identifier.

    Supported rule keys are regex, min_id, max_id, and force_kind.  A rule is
    selected when all supplied conditions match.  This is useful for mixed bulk +
    surface datasets whose filenames encode the family, e.g. numeric slab IDs
    starting at 1000.
    """
    if not rules:
        return None
    value = _numeric_identifier(identifier)
    text = str(identifier)
    for rule in rules:
        kind = rule.get("force_kind") or rule.get("kind")
        if not kind:
            continue
        if "regex" in rule and re.search(str(rule["regex"]), text) is None:
            continue
        if "min_id" in rule and (value is None or value < int(rule["min_id"])):
            continue
        if "max_id" in rule and (value is None or value > int(rule["max_id"])):
            continue
        return str(kind)
    return None


def _pkg_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _column_scale(matrix: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(matrix), axis=0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return scale


def _fit_ridge_original_units(
    matrix: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, Ridge, np.ndarray]:
    scale = _column_scale(matrix)
    estimator = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr", tol=1e-10)
    estimator.fit(matrix / scale, target)
    coefficients = np.asarray(estimator.coef_, dtype=float) / scale
    return coefficients, estimator, scale


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(residual))))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    if denominator < 1e-20:
        return None
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denominator)


def _metric_block(
    target: np.ndarray,
    prediction: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "n": int(len(target)),
        "mae": _mae(target, prediction),
        "rmse": _rmse(target, prediction),
        "r2": _r2(target, prediction),
    }
    by_family: dict[str, Any] = {}
    for family in sorted(set(str(x) for x in families)):
        mask = families == family
        by_family[family] = {
            "n": int(np.count_nonzero(mask)),
            "mae": _mae(target[mask], prediction[mask]),
            "rmse": _rmse(target[mask], prediction[mask]),
            "r2": _r2(target[mask], prediction[mask]),
        }
    block["by_family"] = by_family
    return block


def _make_cv_splits(
    indices: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(indices) < 4 or folds < 2:
        return []
    unique, counts = np.unique(labels[indices], return_counts=True)
    if len(unique) > 1 and int(np.min(counts)) >= 2:
        n_splits = min(int(folds), int(np.min(counts)))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (indices[train_local], indices[val_local])
            for train_local, val_local in splitter.split(indices, labels[indices])
        ]
    n_splits = min(int(folds), len(indices))
    if n_splits < 2:
        return []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (indices[train_local], indices[val_local])
        for train_local, val_local in splitter.split(indices)
    ]


def _select_alpha(
    matrix: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    alphas: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    splits = _make_cv_splits(train_indices, labels, folds, seed)
    if not splits:
        alpha = float(alphas[len(alphas) // 2])
        return alpha, [{"alpha": alpha, "mae_mean": math.nan, "mae_std": math.nan}]
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        scores = []
        for fit_idx, val_idx in splits:
            coefficients, _, _ = _fit_ridge_original_units(
                matrix[fit_idx], target[fit_idx], float(alpha)
            )
            scores.append(_mae(target[val_idx], matrix[val_idx] @ coefficients))
        rows.append(
            {
                "alpha": float(alpha),
                "mae_mean": float(np.mean(scores)),
                "mae_std": float(np.std(scores)),
            }
        )
    best = min(rows, key=lambda row: (row["mae_mean"], row["alpha"]))
    return float(best["alpha"]), rows


@dataclass
class CEModel:
    feature_spec: FeatureSpec
    coefficients: np.ndarray
    mapper_config: MapperConfig
    target_normalization: str
    energy_unit: str
    metadata: dict[str, Any]

    @property
    def cluster_spec(self) -> ClusterSpec:
        return self.feature_spec.cluster_spec

    @property
    def coefficient_tables(self) -> CoefficientTables:
        return self.feature_spec.coefficient_tables(self.coefficients)

    def predict_mapped(
        self,
        mapped: MappedStructure,
        topology: Topology | None = None,
    ) -> dict[str, Any]:
        topology = topology or build_topology(mapped, self.cluster_spec)
        counts, occupations = self.feature_spec.raw_counts(mapped.symbols, topology)
        total_energy = float(np.dot(counts, self.coefficients))
        n_metal = int(np.count_nonzero(occupations != self.feature_spec.vacancy_code))
        denominator = self.feature_spec.denominator(
            self.target_normalization, n_metal, len(occupations)
        )
        return {
            "prediction": total_energy / denominator,
            "total_energy": total_energy,
            "normalization_denominator": denominator,
            "n_metal": n_metal,
            "n_lattice": len(occupations),
            "kind": mapped.kind,
        }

    def predict_path(self, path: str | Path) -> dict[str, Any]:
        mapped = map_structure(path, self.mapper_config)
        result = self.predict_mapped(mapped)
        result["path"] = str(path)
        result["identifier"] = Path(path).stem
        return result

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "coefficients.npz", coefficients=self.coefficients)
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "feature_spec": self.feature_spec.to_dict(),
            "mapper_config": self.mapper_config.to_dict(),
            "target_normalization": self.target_normalization,
            "energy_unit": self.energy_unit,
            "metadata": self.metadata,
        }
        write_json(directory / "metadata.json", payload)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CEModel":
        directory = Path(directory)
        with (directory / "metadata.json").open() as handle:
            payload = json.load(handle)
        if int(payload.get("format_version", -1)) != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format {payload.get('format_version')}; "
                f"expected {MODEL_FORMAT_VERSION}"
            )
        coefficients = np.load(directory / "coefficients.npz")["coefficients"]
        feature_spec = FeatureSpec.from_dict(payload["feature_spec"])
        if len(coefficients) != feature_spec.n_features:
            raise ValueError("Coefficient vector length does not match feature vocabulary")
        return cls(
            feature_spec=feature_spec,
            coefficients=np.asarray(coefficients, dtype=float),
            mapper_config=MapperConfig.from_dict(payload["mapper_config"]),
            target_normalization=payload["target_normalization"],
            energy_unit=payload.get("energy_unit", "eV"),
            metadata=payload.get("metadata", {}),
        )


def train_model(
    config: dict[str, Any],
    progress: Callable[[str], None] | None = print,
) -> tuple[CEModel, pd.DataFrame, dict[str, Any]]:
    data_cfg = dict(config.get("data", {}))
    if "structures" not in data_cfg or "targets" not in data_cfg:
        raise ValueError("Configuration requires data.structures and data.targets")
    strict = bool(data_cfg.get("strict", True))
    extensions = data_cfg.get(
        "extensions", [".cif", ".vasp", ".poscar", ".xyz", ".extxyz"]
    )
    target_normalization = str(data_cfg.get("target_normalization", "per_atom"))
    energy_unit = str(data_cfg.get("energy_unit", "eV"))

    mapper_cfg = MapperConfig.from_dict(config.get("lattice"))
    force_kind_rules = list(data_cfg.get("force_kind_by_identifier", []))
    cluster_spec = ClusterSpec.from_dict(config.get("clusters"))
    regression = dict(config.get("regression", {}))
    seed = int(regression.get("random_seed", 42))
    test_fraction = float(regression.get("test_fraction", 0.20))
    cv_folds = int(regression.get("cv_folds", 5))
    if "alphas" in regression:
        alphas = np.asarray(regression["alphas"], dtype=float)
    else:
        log_min = float(regression.get("alpha_log10_min", -8.0))
        log_max = float(regression.get("alpha_log10_max", 2.0))
        n_alpha = int(regression.get("n_alphas", 31))
        alphas = np.logspace(log_min, log_max, n_alpha)
    if np.any(alphas < 0) or len(alphas) == 0:
        raise ValueError("Ridge alphas must be a non-empty set of nonnegative values")

    targets = read_targets(data_cfg["targets"])
    structures = discover_structures(data_cfg["structures"], extensions)
    records, missing = match_targets(targets, structures, strict=strict)
    if progress:
        progress(
            f"Matched {len(records)} structures to {len(targets)} target rows"
            + (f" ({len(missing)} missing ignored)" if missing else "")
        )

    mapped_list: list[MappedStructure] = []
    topology_list: list[Topology] = []
    topology_cache: dict[str, Topology] = {}
    species: set[str] = set()
    families: list[str] = []
    for index, record in enumerate(records, 1):
        record_mapper_cfg = mapper_cfg
        record_force_kind = _force_kind_for_identifier(record.identifier, force_kind_rules)
        if record_force_kind is not None:
            record_mapper_cfg = replace(mapper_cfg, force_kind=record_force_kind)
        mapped = map_structure(record.path, record_mapper_cfg)
        signature = topology_signature(mapped)
        topology = topology_cache.get(signature)
        if topology is None:
            topology = build_topology(mapped, cluster_spec)
            topology_cache[signature] = topology
        mapped_list.append(mapped)
        topology_list.append(topology)
        species.update(s for s in mapped.symbols if s != mapper_cfg.vacancy_symbol)
        families.append(mapped.kind)
        if progress and (index == 1 or index % 100 == 0 or index == len(records)):
            progress(f"Mapped {index}/{len(records)} structures; topology classes={len(topology_cache)}")

    geometries = sorted(set(g for topology in topology_cache.values() for g in topology.geometry_keys))
    feature_spec = FeatureSpec(
        species=species,
        vacancy_symbol=mapper_cfg.vacancy_symbol,
        cluster_spec=cluster_spec,
        geometries=geometries,
    )
    if progress:
        progress(
            f"Feature space: {feature_spec.n_features} terms, species={feature_spec.species}, "
            f"triplet geometries={geometries}"
        )

    matrix = np.zeros((len(records), feature_spec.n_features), dtype=float)
    target = np.asarray([r.target for r in records], dtype=float)
    denominators = np.zeros(len(records), dtype=float)
    rows: list[dict[str, Any]] = []
    for row_index, (record, mapped, topology) in enumerate(
        zip(records, mapped_list, topology_list)
    ):
        vector, occupations, denominator = feature_spec.vector(
            mapped.symbols, topology, target_normalization
        )
        matrix[row_index] = vector
        denominators[row_index] = denominator
        rows.append(
            {
                "identifier": record.identifier,
                "path": str(record.path),
                "target": record.target,
                "kind": mapped.kind,
                "n_metal": mapped.n_metal,
                "n_lattice": mapped.n_lattice,
                "normalization_denominator": denominator,
            }
        )

    labels = np.asarray(families, dtype=object)
    all_indices = np.arange(len(records), dtype=int)
    if len(records) >= 10 and test_fraction > 0:
        unique, counts = np.unique(labels, return_counts=True)
        stratify = labels if len(unique) > 1 and np.min(counts) >= 2 else None
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify,
        )
        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)
    else:
        train_indices = all_indices
        test_indices = np.empty(0, dtype=int)

    best_alpha, cv_rows = _select_alpha(
        matrix, target, labels, train_indices, alphas, cv_folds, seed
    )
    if progress:
        finite_cv = [r for r in cv_rows if np.isfinite(r["mae_mean"])]
        if finite_cv:
            best_row = min(finite_cv, key=lambda r: r["mae_mean"])
            progress(
                f"Selected ridge alpha={best_alpha:.6g}; CV MAE={best_row['mae_mean']:.6g} {energy_unit}"
            )
        else:
            progress(f"Selected default ridge alpha={best_alpha:.6g} (dataset too small for CV)")

    holdout_coefficients, _, _ = _fit_ridge_original_units(
        matrix[train_indices], target[train_indices], best_alpha
    )
    train_prediction = matrix[train_indices] @ holdout_coefficients
    metrics: dict[str, Any] = {
        "selected_alpha": best_alpha,
        "cross_validation": cv_rows,
        "training_split": _metric_block(
            target[train_indices], train_prediction, labels[train_indices]
        ),
    }
    if len(test_indices):
        test_prediction = matrix[test_indices] @ holdout_coefficients
        metrics["holdout"] = _metric_block(
            target[test_indices], test_prediction, labels[test_indices]
        )
    else:
        metrics["holdout"] = None

    # Refit on all reference structures after estimating generalization error.
    coefficients, _, scale = _fit_ridge_original_units(matrix, target, best_alpha)
    prediction = matrix @ coefficients
    metrics["full_refit"] = _metric_block(target, prediction, labels)
    metrics["n_samples"] = int(len(records))
    metrics["n_features"] = int(feature_spec.n_features)
    metrics["n_topology_classes"] = int(len(topology_cache))
    metrics["n_nonzero_coefficients_1e-12"] = int(np.count_nonzero(np.abs(coefficients) > 1e-12))
    metrics["missing_target_structures"] = missing

    package_versions = {
        name: _pkg_version(name)
        for name in ["numpy", "scipy", "scikit-learn", "ase", "numba", "pandas"]
    }
    metadata = {
        "metrics": metrics,
        "training_config": config,
        "package_versions": package_versions,
        "feature_scale_summary": {
            "min": float(np.min(scale)),
            "max": float(np.max(scale)),
        },
    }
    model = CEModel(
        feature_spec=feature_spec,
        coefficients=coefficients,
        mapper_config=mapper_cfg,
        target_normalization=target_normalization,
        energy_unit=energy_unit,
        metadata=metadata,
    )

    predictions = pd.DataFrame(rows)
    predictions["prediction"] = prediction
    predictions["residual"] = predictions["target"] - predictions["prediction"]
    predictions["absolute_error"] = np.abs(predictions["residual"])
    validation_prediction = matrix @ holdout_coefficients
    predictions["validation_prediction"] = validation_prediction
    predictions["validation_residual"] = predictions["target"] - predictions["validation_prediction"]
    predictions["validation_absolute_error"] = np.abs(predictions["validation_residual"])
    predictions["split"] = "fit"
    if len(test_indices):
        predictions.loc[test_indices, "split"] = "holdout"

    return model, predictions, metrics


def save_training_outputs(
    model: CEModel,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model")
    predictions.to_csv(output / "training_predictions.csv", index=False)
    write_json(output / "metrics.json", metrics)
    coefficient_table = pd.DataFrame(
        {
            "feature": model.feature_spec.feature_names,
            "coefficient": model.coefficients,
            "abs_coefficient": np.abs(model.coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefficient_table.to_csv(output / "coefficients.csv", index=False)
    return output
