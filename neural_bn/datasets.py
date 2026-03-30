"""Dataset adapters and benchmark suite definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

from .synthetic import SyntheticDataset, make_mixed_classification_dataset


@dataclass(slots=True)
class DatasetArtifact:
    name: str
    regime: str
    frame: pd.DataFrame
    target_column: str
    truth_graph: nx.DiGraph | None = None
    source: str = "builtin"
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetSplit:
    artifact: DatasetArtifact
    split_id: str
    seed: int
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    split_metadata: Dict[str, float]


class DatasetAdapter:
    name: str
    regime: str

    def is_available(self) -> tuple[bool, str]:
        return True, "available"

    def load(self) -> DatasetArtifact:
        raise NotImplementedError

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        raise NotImplementedError


class InMemoryDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        artifact: DatasetArtifact,
        repeats: int = 1,
        test_size: float = 0.2,
        validation_size: float = 0.2,
    ) -> None:
        self.name = artifact.name
        self.regime = artifact.regime
        self.artifact = artifact
        self.repeats = repeats
        self.test_size = test_size
        self.validation_size = validation_size

    def load(self) -> DatasetArtifact:
        return self.artifact

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        artifact = self.load()
        y = artifact.frame[artifact.target_column]
        for repeat_idx in range(self.repeats):
            repeat_seed = seed + repeat_idx
            train_val, test = train_test_split(
                artifact.frame,
                test_size=self.test_size,
                random_state=repeat_seed,
                stratify=y if artifact.regime == "real" else None,
            )
            train, validation = train_test_split(
                train_val,
                test_size=self.validation_size,
                random_state=repeat_seed,
                stratify=train_val[artifact.target_column] if artifact.regime == "real" else None,
            )
            yield DatasetSplit(
                artifact=artifact,
                split_id=f"repeat_{repeat_idx}",
                seed=repeat_seed,
                train_frame=train.reset_index(drop=True),
                validation_frame=validation.reset_index(drop=True),
                test_frame=test.reset_index(drop=True),
                split_metadata={
                    "train_size": float(len(train)),
                    "validation_size": float(len(validation)),
                    "test_size": float(len(test)),
                },
            )


class SyntheticStructureDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        name: str,
        n_samples: int,
        n_features: int,
        repeats: int = 2,
        generator: Callable[..., SyntheticDataset] = make_mixed_classification_dataset,
    ) -> None:
        self.name = name
        self.regime = "known_graph"
        self.n_samples = n_samples
        self.n_features = n_features
        self.repeats = repeats
        self.generator = generator

    def load(self, seed: int | None = None) -> DatasetArtifact:
        dataset = self.generator(
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=seed or 7,
        )
        return DatasetArtifact(
            name=self.name,
            regime=self.regime,
            frame=dataset.frame,
            target_column=dataset.target_column,
            truth_graph=dataset.graph,
            source="synthetic",
        )

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        for repeat_idx in range(self.repeats):
            repeat_seed = seed + repeat_idx
            artifact = self.load(seed=repeat_seed)
            train_val, test = train_test_split(
                artifact.frame,
                test_size=0.25,
                random_state=repeat_seed,
            )
            train, validation = train_test_split(
                train_val,
                test_size=0.2,
                random_state=repeat_seed,
            )
            yield DatasetSplit(
                artifact=artifact,
                split_id=f"repeat_{repeat_idx}",
                seed=repeat_seed,
                train_frame=train.reset_index(drop=True),
                validation_frame=validation.reset_index(drop=True),
                test_frame=test.reset_index(drop=True),
                split_metadata={
                    "train_size": float(len(train)),
                    "validation_size": float(len(validation)),
                    "test_size": float(len(test)),
                },
            )


class SklearnClassificationDatasetAdapter(DatasetAdapter):
    LOADERS: Dict[str, Callable[[], object]] = {
        "sklearn_breast_cancer": load_breast_cancer,
        "sklearn_iris": load_iris,
        "sklearn_wine": load_wine,
    }

    def __init__(self, name: str, repeats: int = 2) -> None:
        self.name = name
        self.regime = "real"
        self.repeats = repeats

    def load(self) -> DatasetArtifact:
        bunch = self.LOADERS[self.name](as_frame=True)
        frame = bunch.frame.copy()
        target_column = bunch.target.name if bunch.target.name else "target"
        if target_column not in frame.columns:
            frame[target_column] = bunch.target
        if frame[target_column].dtype.kind in {"i", "u", "f"}:
            frame[target_column] = frame[target_column].astype(str)
        return DatasetArtifact(
            name=self.name,
            regime=self.regime,
            frame=frame,
            target_column=target_column,
            source="sklearn",
        )

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        artifact = self.load()
        in_memory = InMemoryDatasetAdapter(artifact, repeats=self.repeats)
        yield from in_memory.iter_splits(seed)


class CsvClassificationDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        csv_path: Path,
        target_column: str,
        name: str | None = None,
        repeats: int = 1,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target_column = target_column
        self.name = name or self.csv_path.stem
        self.regime = "real"
        self.repeats = repeats

    def is_available(self) -> tuple[bool, str]:
        return self.csv_path.exists(), "available" if self.csv_path.exists() else "csv file missing"

    def load(self) -> DatasetArtifact:
        frame = pd.read_csv(self.csv_path)
        return DatasetArtifact(
            name=self.name,
            regime=self.regime,
            frame=frame,
            target_column=self.target_column,
            source=str(self.csv_path),
        )

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        artifact = self.load()
        in_memory = InMemoryDatasetAdapter(artifact, repeats=self.repeats)
        yield from in_memory.iter_splits(seed)


class LocalStructuralDatasetAdapter(DatasetAdapter):
    def __init__(self, root: Path, name: str | None = None, repeats: int = 2) -> None:
        self.root = Path(root)
        self.name = name or self.root.name
        self.regime = "known_graph"
        self.repeats = repeats

    def is_available(self) -> tuple[bool, str]:
        frame_path = self.root / "frame.csv"
        edges_path = self.root / "edges.csv"
        exists = frame_path.exists() and edges_path.exists()
        detail = "available" if exists else "expected frame.csv and edges.csv"
        return exists, detail

    def load(self) -> DatasetArtifact:
        frame = pd.read_csv(self.root / "frame.csv")
        edges = pd.read_csv(self.root / "edges.csv")
        metadata = self._load_metadata()
        target_column = metadata.get("target_column", "label" if "label" in frame.columns else frame.columns[-1])
        graph = nx.DiGraph()
        graph.add_nodes_from(frame.columns)
        for row in edges.itertuples(index=False):
            graph.add_edge(str(row.source), str(row.target))
        return DatasetArtifact(
            name=self.name,
            regime=self.regime,
            frame=frame,
            target_column=target_column,
            truth_graph=graph,
            source=str(metadata.get("source", self.root)),
            notes=[str(note) for note in metadata.get("notes", [])],
        )

    def iter_splits(self, seed: int) -> Iterator[DatasetSplit]:
        artifact = self.load()
        in_memory = InMemoryDatasetAdapter(artifact, repeats=self.repeats)
        yield from in_memory.iter_splits(seed)

    def _load_metadata(self) -> dict:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text())


@dataclass(slots=True)
class BenchmarkSuiteDefinition:
    name: str
    datasets: List[str]
    baselines: List[str]
    seeds: List[int]
    ablations: List[str]


def dataset_registry(base_dir: Path | None = None) -> Dict[str, DatasetAdapter]:
    base = Path(base_dir or ".")
    expert_root = base / "datasets" / "expert_bn"
    return {
        "synthetic_tiny": SyntheticStructureDatasetAdapter("synthetic_tiny", n_samples=180, n_features=5, repeats=1),
        "synthetic_small": SyntheticStructureDatasetAdapter("synthetic_small", n_samples=300, n_features=6, repeats=2),
        "synthetic_medium": SyntheticStructureDatasetAdapter("synthetic_medium", n_samples=500, n_features=8, repeats=2),
        "sklearn_breast_cancer": SklearnClassificationDatasetAdapter("sklearn_breast_cancer", repeats=2),
        "sklearn_wine": SklearnClassificationDatasetAdapter("sklearn_wine", repeats=2),
        "sklearn_iris": SklearnClassificationDatasetAdapter("sklearn_iris", repeats=2),
        "alarm": LocalStructuralDatasetAdapter(expert_root / "alarm", name="alarm", repeats=2),
        "child": LocalStructuralDatasetAdapter(expert_root / "child", name="child", repeats=2),
        "insurance": LocalStructuralDatasetAdapter(expert_root / "insurance", name="insurance", repeats=2),
    }


def benchmark_suites() -> Dict[str, BenchmarkSuiteDefinition]:
    return {
        "smoke": BenchmarkSuiteDefinition(
            name="smoke",
            datasets=["synthetic_tiny", "sklearn_breast_cancer"],
            baselines=["neural_screened_bn", "greedy_hc_bn", "naive_bayes", "tan"],
            seeds=[7],
            ablations=["default", "no_screener"],
        ),
        "paper_minimum": BenchmarkSuiteDefinition(
            name="paper_minimum",
            datasets=[
                "synthetic_small",
                "synthetic_medium",
                "alarm",
                "child",
                "insurance",
                "sklearn_breast_cancer",
                "sklearn_wine",
            ],
            baselines=[
                "neural_screened_bn",
                "greedy_hc_bn",
                "naive_bayes",
                "tan",
                "kdb",
                "aode",
                "xgboost",
                "lightgbm",
                "catboost",
                "tabpfn",
            ],
            seeds=[7, 13],
            ablations=[
                "default",
                "no_screener",
                "no_bundles",
                "no_bootstrap",
                "generative_only",
                "discriminative_only",
                "small_pool",
                "wide_parents",
            ],
        ),
    }
