"""Vacancy-aware fcc cluster expansion and Monte Carlo annealing."""

from .features import FeatureSpec
from .lattice import MapperConfig, MappedStructure, map_atoms, map_structure
from .mc import replay_swap_history, run_annealing
from .model import CEModel, train_model
from .topology import ClusterSpec, build_topology

__all__ = [
    "CEModel",
    "ClusterSpec",
    "FeatureSpec",
    "MappedStructure",
    "MapperConfig",
    "build_topology",
    "map_atoms",
    "map_structure",
    "replay_swap_history",
    "run_annealing",
    "train_model",
]

__version__ = "0.1.2"
