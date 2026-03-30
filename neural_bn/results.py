"""Serializable experiment result objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class ExperimentMetadata:
    model_name: str
    dataset_name: str = "in_memory"
    dataset_regime: str = "unknown"
    split_id: str = "fit"
    seed: int | None = None
    ablation_label: str = "default"
    baseline_family: str = "method"


@dataclass(slots=True)
class ExperimentResult:
    metadata: ExperimentMetadata
    predictive: Dict[str, float]
    structural: Dict[str, float] | None
    search_stats: Dict[str, Any]
    runtimes: Dict[str, float]
    split_metadata: Dict[str, float] = field(default_factory=dict)
    graph_edges: List[tuple[str, str]] = field(default_factory=list)
    candidate_pools: Dict[str, List[str]] = field(default_factory=dict)
    screener_diagnostics: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    notes: List[str] = field(default_factory=list)
    screener: Any | None = None
    search: Any | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "predictive": self.predictive,
            "structural": self.structural,
            "search_stats": self.search_stats,
            "runtimes": self.runtimes,
            "split_metadata": self.split_metadata,
            "graph_edges": [list(edge) for edge in self.graph_edges],
            "candidate_pools": self.candidate_pools,
            "screener_diagnostics": self.screener_diagnostics,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(slots=True)
class BenchmarkSuiteResult:
    suite_name: str
    records: List[ExperimentResult]
    tables: Dict[str, List[Dict[str, Any]]]
    run_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "records": [record.to_dict() for record in self.records],
            "tables": self.tables,
            "run_metadata": self.run_metadata,
        }
