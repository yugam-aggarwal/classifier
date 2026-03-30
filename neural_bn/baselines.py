"""Baseline registry with built-in BN baselines and optional external wrappers."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .models import AODEClassifier, DiscreteBayesianNetwork, kdb_graph, naive_bayes_graph, tan_graph


@dataclass(slots=True)
class BaselineAvailability:
    name: str
    available: bool
    detail: str


class NaiveBayesBNClassifier:
    def __init__(self, target_column: str, laplace: float = 1.0) -> None:
        self.target_column = target_column
        self.laplace = laplace
        self.model_: DiscreteBayesianNetwork | None = None

    def fit(self, frame: pd.DataFrame) -> "NaiveBayesBNClassifier":
        graph = naive_bayes_graph(frame.columns, self.target_column)
        self.model_ = DiscreteBayesianNetwork(self.target_column, self.laplace).fit(frame, graph)
        return self

    def predict_proba(self, frame: pd.DataFrame):
        return self.model_.predict_proba(frame)


class TANClassifier:
    def __init__(self, target_column: str, laplace: float = 1.0) -> None:
        self.target_column = target_column
        self.laplace = laplace
        self.model_: DiscreteBayesianNetwork | None = None

    def fit(self, frame: pd.DataFrame) -> "TANClassifier":
        graph = tan_graph(frame, self.target_column)
        self.model_ = DiscreteBayesianNetwork(self.target_column, self.laplace).fit(frame, graph)
        return self

    def predict_proba(self, frame: pd.DataFrame):
        return self.model_.predict_proba(frame)


class KDBClassifier:
    def __init__(self, target_column: str, k: int = 2, laplace: float = 1.0) -> None:
        self.target_column = target_column
        self.k = k
        self.laplace = laplace
        self.model_: DiscreteBayesianNetwork | None = None

    def fit(self, frame: pd.DataFrame) -> "KDBClassifier":
        graph = kdb_graph(frame, self.target_column, self.k)
        self.model_ = DiscreteBayesianNetwork(self.target_column, self.laplace).fit(frame, graph)
        return self

    def predict_proba(self, frame: pd.DataFrame):
        return self.model_.predict_proba(frame)


def discover_tabpfn_checkpoint() -> Path | None:
    configured_path = os.environ.get("NEURAL_BN_TABPFN_MODEL_PATH")
    if configured_path:
        candidate = Path(configured_path).expanduser()
        return candidate if candidate.exists() else None

    search_roots = []
    configured_cache = os.environ.get("NEURAL_BN_TABPFN_CACHE_DIR")
    if configured_cache:
        search_roots.append(Path(configured_cache).expanduser())
    search_roots.extend(
        [
            Path.home() / ".cache" / "tabpfn",
            Path("/tmp/neural_bn_tabpfn"),
        ]
    )

    candidates: list[Path] = []
    seen_roots: set[Path] = set()
    for root in search_roots:
        expanded = root.expanduser()
        if expanded in seen_roots or not expanded.exists():
            continue
        seen_roots.add(expanded)
        if expanded.is_file():
            candidates.append(expanded)
            continue
        for pattern in ("tabpfn*.ckpt", "*.ckpt", "*.pth", "*.pt"):
            candidates.extend(path for path in expanded.glob(pattern) if path.is_file())

    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def tabpfn_availability_detail() -> str:
    configured_path = os.environ.get("NEURAL_BN_TABPFN_MODEL_PATH")
    if configured_path:
        resolved = Path(configured_path).expanduser()
        if resolved.exists():
            return f"installed; configured checkpoint found at {resolved}"
        return f"installed; configured checkpoint missing at {resolved}"

    checkpoint = discover_tabpfn_checkpoint()
    if checkpoint is not None:
        return f"installed; local checkpoint found at {checkpoint}"
    return "installed; no local checkpoint discovered"


def external_baseline_availability() -> Dict[str, BaselineAvailability]:
    checks = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "tabpfn": "tabpfn",
    }
    availability: Dict[str, BaselineAvailability] = {}
    for name, module_name in checks.items():
        available = importlib.util.find_spec(module_name) is not None
        detail = (
            tabpfn_availability_detail()
            if available and name == "tabpfn"
            else "installed" if available else "not installed"
        )
        availability[name] = BaselineAvailability(name=name, available=available, detail=detail)
    availability["tabm"] = BaselineAvailability(
        name="tabm",
        available=False,
        detail="architecture not bundled; integrate separately if needed",
    )
    availability["ft_transformer"] = BaselineAvailability(
        name="ft_transformer",
        available=False,
        detail="architecture not bundled; integrate separately if needed",
    )
    availability["tabicl"] = BaselineAvailability(
        name="tabicl",
        available=False,
        detail="foundation model not bundled; integrate separately if needed",
    )
    return availability
