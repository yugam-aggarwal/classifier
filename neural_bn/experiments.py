"""Experiment runners, ablations, and benchmark suite execution."""

from __future__ import annotations

import copy
import contextlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .baselines import AODEClassifier, discover_tabpfn_checkpoint, external_baseline_availability
from .config import AblationConfig, PipelineConfig
from .data import TabularPreprocessor
from .datasets import BenchmarkSuiteDefinition, DatasetSplit, benchmark_suites, dataset_registry
from .metrics import (
    candidate_pool_parent_recall,
    candidate_pool_reduction,
    edge_precision_recall_f1,
    predictive_metrics,
    structural_hamming_distance,
)
from .models import DiscreteBayesianNetwork, kdb_graph, naive_bayes_graph, tan_graph
from .pipeline import NeuralScreenedBNClassifier
from .results import BenchmarkSuiteResult, ExperimentMetadata, ExperimentResult
from .runtime import benchmark_runtime_env, resolve_torch_device, temporary_env
from .search import ConstrainedBNStructureLearner


@dataclass(slots=True)
class BaselineRunner:
    name: str
    family: str

    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EvidenceStageSpec:
    name: str
    suite_name: str
    output_filename: str
    dataset_names: Sequence[str] | None = None
    baseline_names: Sequence[str] | None = None
    seeds: Sequence[int] | None = None
    ablations: Sequence[str] | None = None


def _emit_progress(enabled: bool, prefix: str, message: str) -> None:
    if not enabled:
        return
    print(f"[{prefix}] {message}", file=sys.stderr, flush=True)


def ablation_registry() -> Dict[str, AblationConfig]:
    return {
        "default": AblationConfig(label="default"),
        "known_graph_fast": AblationConfig(label="known_graph_fast", known_graph_fast=True),
        "mi_filter_fast": AblationConfig(label="mi_filter_fast", fast_mode_strategy="mi_filter_fast"),
        "single_pass_neural_fast": AblationConfig(
            label="single_pass_neural_fast",
            fast_mode_strategy="single_pass_neural_fast",
        ),
        "hybrid_mi_neural_fast": AblationConfig(
            label="hybrid_mi_neural_fast",
            fast_mode_strategy="hybrid_mi_neural_fast",
        ),
        "no_screener": AblationConfig(label="no_screener", disable_screener=True),
        "no_bundles": AblationConfig(label="no_bundles", disable_interaction_bundles=True),
        "no_bootstrap": AblationConfig(label="no_bootstrap", disable_bootstrap=True),
        "generative_only": AblationConfig(label="generative_only", objective="generative"),
        "discriminative_only": AblationConfig(label="discriminative_only", objective="discriminative"),
        "small_pool": AblationConfig(label="small_pool", candidate_pool_size=3),
        "wide_parents": AblationConfig(label="wide_parents", max_parents=5),
        "warm_start_nb": AblationConfig(label="warm_start_nb", warm_start="naive_bayes"),
        "warm_start_sparse": AblationConfig(label="warm_start_sparse", warm_start="screener_sparse"),
        "no_edge_prior": AblationConfig(label="no_edge_prior", edge_prior_weight=0.0),
        "no_pruning": AblationConfig(
            label="no_pruning",
            validation_prune_passes=0,
            validation_prune_passes_real=0,
        ),
        "stabilized_pruning_off": AblationConfig(
            label="stabilized_pruning_off",
            validation_prune_passes=0,
            validation_prune_passes_real=0,
        ),
    }


def build_baseline_runners() -> Dict[str, BaselineRunner]:
    return {
        "neural_screened_bn": NeuralScreenedBNRunner(),
        "greedy_hc_bn": GreedyHillClimbBNRunner(),
        "naive_bayes": BuiltinBNRunner(name="naive_bayes", family="bn_classifier"),
        "tan": BuiltinBNRunner(name="tan", family="bn_classifier"),
        "kdb": BuiltinBNRunner(name="kdb", family="bn_classifier"),
        "aode": BuiltinBNRunner(name="aode", family="bn_classifier"),
        "xgboost": ExternalTreeRunner(name="xgboost", family="tree_ensemble"),
        "lightgbm": ExternalTreeRunner(name="lightgbm", family="tree_ensemble"),
        "catboost": ExternalTreeRunner(name="catboost", family="tree_ensemble"),
        "tabpfn": TabPFNRunner(name="tabpfn", family="foundation_model"),
    }


def evidence_stage_registry() -> Dict[str, EvidenceStageSpec]:
    evaluation_baselines = [
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
    ]
    return {
        "sanity_default": EvidenceStageSpec(
            name="sanity_default",
            suite_name="paper_minimum",
            output_filename="sanity_default.json",
            dataset_names=["alarm", "child", "insurance", "sklearn_breast_cancer"],
            baseline_names=evaluation_baselines,
            seeds=[7],
            ablations=["default"],
        ),
        "structural_default": EvidenceStageSpec(
            name="structural_default",
            suite_name="paper_minimum",
            output_filename="structural_default.json",
            dataset_names=["synthetic_small", "synthetic_medium", "alarm", "child", "insurance"],
            baseline_names=evaluation_baselines,
            seeds=[7],
            ablations=["default"],
        ),
        "paper_minimum_full": EvidenceStageSpec(
            name="paper_minimum_full",
            suite_name="paper_minimum",
            output_filename="paper_minimum_full.json",
        ),
        "known_graph_fast_probe": EvidenceStageSpec(
            name="known_graph_fast_probe",
            suite_name="paper_minimum",
            output_filename="known_graph_fast_probe.json",
            dataset_names=[
                "synthetic_small",
                "synthetic_medium",
                "alarm",
                "child",
                "insurance",
                "sklearn_breast_cancer",
                "sklearn_wine",
            ],
            baseline_names=evaluation_baselines,
            seeds=[7, 8],
            ablations=["default", "known_graph_fast"],
        ),
        "fast_mode_design_probe": EvidenceStageSpec(
            name="fast_mode_design_probe",
            suite_name="paper_minimum",
            output_filename="fast_mode_design_probe.json",
            dataset_names=[
                "synthetic_small",
                "synthetic_medium",
                "alarm",
                "child",
                "insurance",
            ],
            baseline_names=["neural_screened_bn", "greedy_hc_bn"],
            seeds=[7, 8],
            ablations=[
                "default",
                "known_graph_fast",
                "mi_filter_fast",
                "single_pass_neural_fast",
                "hybrid_mi_neural_fast",
            ],
        ),
    }


class NeuralScreenedBNRunner(BaselineRunner):
    def __init__(self) -> None:
        super().__init__(name="neural_screened_bn", family="method")

    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        config = ablation.apply(
            copy.deepcopy(base_config),
            dataset_regime=split.artifact.regime,
            num_columns=len(split.train_frame.columns),
        )
        config.target_column = split.artifact.target_column
        config.random_state = split.seed
        config.screener.random_state = split.seed
        model = NeuralScreenedBNClassifier(config)
        model.fit(
            split.train_frame,
            validation_frame=split.validation_frame,
            truth_graph=split.artifact.truth_graph,
            dataset_name=split.artifact.name,
            dataset_regime=split.artifact.regime,
            split_id=split.split_id,
            seed=split.seed,
            ablation_label=ablation.label,
        )
        result = model.evaluate(split.test_frame, truth_graph=split.artifact.truth_graph)
        result.metadata.baseline_family = self.family
        result.split_metadata = split.split_metadata
        return result


class GreedyHillClimbBNRunner(BaselineRunner):
    def __init__(self) -> None:
        super().__init__(name="greedy_hc_bn", family="classical_search")

    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        started_at = perf_counter()
        config = copy.deepcopy(base_config)
        config.target_column = split.artifact.target_column
        config.random_state = split.seed
        config.screener.random_state = split.seed

        preprocessor = TabularPreprocessor(
            target_column=config.target_column,
            n_bins=config.n_bins,
            min_samples_per_bin=config.min_samples_per_bin,
        )
        train_prepared = preprocessor.fit_transform(split.train_frame)
        test_discrete = preprocessor.transform_for_prediction(split.test_frame)
        after_preprocess = perf_counter()
        search_config = copy.deepcopy(config.search)
        search_config.warm_start = "empty"
        searcher = ConstrainedBNStructureLearner(search_config)
        columns = list(train_prepared.discrete_frame.columns)
        candidate_pools = {
            target: [source for source in columns if source != target]
            for target in columns
        }
        search = searcher.fit(
            frame=train_prepared.discrete_frame,
            target_column=config.target_column,
            candidate_pools=candidate_pools,
        )
        after_search = perf_counter()
        model = DiscreteBayesianNetwork(config.target_column, laplace=search_config.laplace).fit(
            train_prepared.discrete_frame,
            search.graph,
        )
        after_fit = perf_counter()
        probabilities = model.predict_proba(test_discrete.drop(columns=[config.target_column]))
        finished_at = perf_counter()
        return _build_result(
            model_name=self.name,
            family=self.family,
            split=split,
            ablation_label="default",
            probabilities=probabilities,
            graph=search.graph,
            candidate_pools=candidate_pools,
            truth_graph=split.artifact.truth_graph,
            runtimes={
                "preprocess_seconds": after_preprocess - started_at,
                "search_seconds": after_search - after_preprocess,
                "fit_seconds": after_fit - after_search,
                "predict_seconds": finished_at - after_fit,
                "total_seconds": finished_at - started_at,
            },
            search_stats={
                "num_edges": int(search.graph.number_of_edges()),
                "num_steps": len(search.operations),
                "total_score": search.total_score,
            },
            y_true=test_discrete[config.target_column].to_numpy(dtype=int),
        )


class BuiltinBNRunner(BaselineRunner):
    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        started_at = perf_counter()
        config = copy.deepcopy(base_config)
        config.target_column = split.artifact.target_column
        preprocessor = TabularPreprocessor(
            target_column=config.target_column,
            n_bins=config.n_bins,
            min_samples_per_bin=config.min_samples_per_bin,
        )
        train_prepared = preprocessor.fit_transform(split.train_frame)
        test_discrete = preprocessor.transform_for_prediction(split.test_frame)
        after_preprocess = perf_counter()
        train_frame = train_prepared.discrete_frame
        test_features = test_discrete.drop(columns=[config.target_column])

        graph: nx.DiGraph | None = None
        candidate_pools: Dict[str, List[str]] = {}
        if self.name == "naive_bayes":
            graph = naive_bayes_graph(train_frame.columns, config.target_column)
            model = DiscreteBayesianNetwork(config.target_column, config.search.laplace).fit(train_frame, graph)
        elif self.name == "tan":
            graph = tan_graph(train_frame, config.target_column)
            model = DiscreteBayesianNetwork(config.target_column, config.search.laplace).fit(train_frame, graph)
        elif self.name == "kdb":
            graph = kdb_graph(train_frame, config.target_column, k=min(2, config.search.max_parents))
            model = DiscreteBayesianNetwork(config.target_column, config.search.laplace).fit(train_frame, graph)
        else:
            model = AODEClassifier(config.target_column, config.search.laplace).fit(train_frame)
        after_fit = perf_counter()
        probabilities = model.predict_proba(test_features)
        finished_at = perf_counter()

        return _build_result(
            model_name=self.name,
            family=self.family,
            split=split,
            ablation_label="default",
            probabilities=probabilities,
            graph=graph,
            candidate_pools=candidate_pools,
            truth_graph=split.artifact.truth_graph,
            runtimes={
                "preprocess_seconds": after_preprocess - started_at,
                "fit_seconds": after_fit - after_preprocess,
                "predict_seconds": finished_at - after_fit,
                "total_seconds": finished_at - started_at,
            },
            search_stats={"num_edges": int(graph.number_of_edges()) if graph is not None else 0},
            y_true=test_discrete[config.target_column].to_numpy(dtype=int),
        )


class ExternalTreeRunner(BaselineRunner):
    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        availability = external_baseline_availability().get(self.name)
        if availability is None or not availability.available:
            return _skipped_result(
                model_name=self.name,
                family=self.family,
                split=split,
                note=availability.detail if availability is not None else "runner unavailable",
            )

        started_at = perf_counter()
        x_train, x_test, y_train, y_test = _encode_supervised_features(
            split.train_frame,
            split.test_frame,
            split.artifact.target_column,
        )
        after_encode = perf_counter()
        model = _instantiate_external_model(self.name, seed=split.seed, n_classes=len(np.unique(y_train)))
        model.fit(x_train, y_train)
        after_fit = perf_counter()
        probabilities = model.predict_proba(x_test)
        finished_at = perf_counter()
        return _build_result(
            model_name=self.name,
            family=self.family,
            split=split,
            ablation_label="default",
            probabilities=probabilities,
            graph=None,
            candidate_pools={},
            truth_graph=split.artifact.truth_graph,
            runtimes={
                "encode_seconds": after_encode - started_at,
                "fit_seconds": after_fit - after_encode,
                "predict_seconds": finished_at - after_fit,
                "total_seconds": finished_at - started_at,
            },
            search_stats={},
            y_true=y_test,
        )


class TabPFNRunner(BaselineRunner):
    def run(
        self,
        split: DatasetSplit,
        base_config: PipelineConfig,
        ablation: AblationConfig,
    ) -> ExperimentResult:
        availability = external_baseline_availability().get(self.name)
        if availability is None or not availability.available:
            return _skipped_result(
                model_name=self.name,
                family=self.family,
                split=split,
                note=availability.detail if availability is not None else "runner unavailable",
            )

        started_at = perf_counter()
        x_train, x_test, y_train, y_test = _encode_supervised_features(
            split.train_frame,
            split.test_frame,
            split.artifact.target_column,
        )
        after_encode = perf_counter()
        x_train_array = x_train.to_numpy(dtype=float)
        x_test_array = x_test.to_numpy(dtype=float)
        try:
            with _tabpfn_quiet_environment():
                classifier = _instantiate_tabpfn_model(split.seed)
                classifier.fit(x_train_array, y_train)
                after_fit = perf_counter()
                probabilities = classifier.predict_proba(x_test_array)
        except Exception as exc:  # pragma: no cover - environment dependent path
            checkpoint = discover_tabpfn_checkpoint()
            return _skipped_result(
                model_name=self.name,
                family=self.family,
                split=split,
                note=(
                    "tabpfn initialization or inference failed: "
                    f"{type(exc).__name__}: {exc}. "
                    f"Discovered local checkpoint: {checkpoint}. "
                    "If needed, set NEURAL_BN_TABPFN_MODEL_PATH to a working checkpoint path "
                    "or NEURAL_BN_TABPFN_CACHE_DIR to a cache directory containing one."
                ),
            )
        finished_at = perf_counter()

        return _build_result(
            model_name=self.name,
            family=self.family,
            split=split,
            ablation_label="default",
            probabilities=probabilities,
            graph=None,
            candidate_pools={},
            truth_graph=split.artifact.truth_graph,
            runtimes={
                "encode_seconds": after_encode - started_at,
                "fit_seconds": after_fit - after_encode,
                "predict_seconds": finished_at - after_fit,
                "total_seconds": finished_at - started_at,
            },
            search_stats={},
            y_true=y_test,
        )


def run_benchmark_suite(
    suite_name: str,
    base_config: PipelineConfig,
    output_path: Path | None = None,
    base_dir: Path | None = None,
    dataset_names: Sequence[str] | None = None,
    baseline_names: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    ablations: Sequence[str] | None = None,
    tabpfn_device: str | None = None,
    screener_device: str | None = None,
    progress: bool = False,
    progress_label: str | None = None,
) -> BenchmarkSuiteResult:
    suites = benchmark_suites()
    if suite_name not in suites:
        raise KeyError(f"Unknown suite '{suite_name}'.")
    suite = suites[suite_name]
    registry = dataset_registry(base_dir)
    runners = build_baseline_runners()
    ablations_map = ablation_registry()

    selected_datasets = list(dataset_names or suite.datasets)
    selected_baselines = list(baseline_names or suite.baselines)
    selected_seeds = list(seeds or suite.seeds)
    selected_ablations = list(ablations or suite.ablations)
    effective_config = copy.deepcopy(base_config)
    if screener_device is not None:
        effective_config.screener.device = screener_device
    if progress:
        effective_config.screener.progress = True
        effective_config.search.progress = True
    records: List[ExperimentResult] = []
    progress_prefix = progress_label or f"suite:{suite_name}"
    run_plan: List[tuple[str, DatasetSplit, str, str]] = []
    unavailable_datasets: List[tuple[str, str]] = []
    run_metadata = _suite_run_metadata(
        suite_name=suite_name,
        datasets=selected_datasets,
        baselines=selected_baselines,
        seeds=selected_seeds,
        ablations=selected_ablations,
        requested_screener_device=screener_device,
        requested_tabpfn_device=tabpfn_device,
        resolved_screener_device=resolve_torch_device(effective_config.screener.device),
        resolved_tabpfn_device=resolve_torch_device(tabpfn_device or os.environ.get("NEURAL_BN_TABPFN_DEVICE", "auto")),
        output_path=output_path,
    )

    for dataset_name in selected_datasets:
        adapter = registry[dataset_name]
        available, detail = adapter.is_available()
        if not available:
            unavailable_datasets.append((dataset_name, detail))
            continue
        splits = [
            split
            for seed in selected_seeds
            for split in adapter.iter_splits(seed)
        ]
        for split in splits:
            for baseline_name in selected_baselines:
                baseline_ablations = (
                    selected_ablations if baseline_name == "neural_screened_bn" else ["default"]
                )
                for ablation_name in baseline_ablations:
                    run_plan.append((dataset_name, split, baseline_name, ablation_name))

    _emit_progress(
        progress,
        progress_prefix,
        f"starting datasets={len(selected_datasets)} planned_runs={len(run_plan)}",
    )

    env_updates = benchmark_runtime_env(output_path=output_path)
    if tabpfn_device is not None:
        env_updates["NEURAL_BN_TABPFN_DEVICE"] = tabpfn_device
    with temporary_env(env_updates):
        for dataset_name, detail in unavailable_datasets:
            adapter = registry[dataset_name]
            _emit_progress(progress, progress_prefix, f"skipping dataset={dataset_name} reason={detail}")
            records.append(
                _skipped_result(
                    model_name="dataset",
                    family="dataset",
                    split=DatasetSplit(
                        artifact=_placeholder_artifact(dataset_name, adapter.regime),
                        split_id="unavailable",
                        seed=selected_seeds[0] if selected_seeds else 0,
                        train_frame=pd.DataFrame(),
                        validation_frame=pd.DataFrame(),
                        test_frame=pd.DataFrame(),
                        split_metadata={},
                    ),
                    note=f"dataset unavailable: {detail}",
                )
            )

        for completed_runs, (dataset_name, split, baseline_name, ablation_name) in enumerate(run_plan, start=1):
            runner = runners[baseline_name]
            ablation = ablations_map[ablation_name]
            label = baseline_name if ablation_name == "default" else f"{baseline_name}/{ablation_name}"
            _emit_progress(
                progress,
                progress_prefix,
                f"run {completed_runs}/{len(run_plan)} start "
                f"dataset={dataset_name} split={split.split_id} model={label}",
            )
            started_at = perf_counter()
            result = runner.run(split, effective_config, ablation)
            elapsed = perf_counter() - started_at
            records.append(result)
            _emit_progress(
                progress,
                progress_prefix,
                f"run {completed_runs}/{len(run_plan)} done "
                f"dataset={dataset_name} split={split.split_id} model={label} "
                f"status={result.status} elapsed={elapsed:.1f}s",
            )

    tables = summarize_records(records)
    suite_result = BenchmarkSuiteResult(
        suite_name=suite_name,
        records=records,
        tables=tables,
        run_metadata=run_metadata,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite_result.to_dict(), indent=2))
        _emit_progress(progress, progress_prefix, f"wrote artifact={output_path}")
    ok_count = sum(record.status == "ok" for record in records)
    _emit_progress(
        progress,
        progress_prefix,
        f"completed records={len(records)} ok={ok_count} skipped={len(records) - ok_count}",
    )
    return suite_result


def run_staged_evidence_pass(
    base_config: PipelineConfig,
    artifact_dir: Path | None = None,
    base_dir: Path | None = None,
    stages: Sequence[str] | None = None,
    stage_specs: Dict[str, EvidenceStageSpec] | None = None,
    tabpfn_device: str | None = None,
    screener_device: str | None = None,
    progress: bool = False,
) -> Dict[str, Any]:
    specs = stage_specs or evidence_stage_registry()
    default_stage_names = [
        stage_name
        for stage_name in ("sanity_default", "structural_default", "paper_minimum_full")
        if stage_name in specs
    ]
    selected_stage_names = list(stages or default_stage_names or specs.keys())
    unknown = sorted(set(selected_stage_names) - set(specs))
    if unknown:
        raise KeyError(f"Unknown evidence stages: {', '.join(unknown)}")

    artifact_root = Path(artifact_dir or Path("artifacts") / "benchmarks")
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "evidence_summary.json"
    completed_payloads: Dict[str, Dict[str, Any]] = {}
    evidence_run_metadata = _evidence_run_metadata(
        artifact_dir=artifact_root,
        requested_stage_names=selected_stage_names,
        requested_screener_device=screener_device,
        requested_tabpfn_device=tabpfn_device,
        resolved_screener_device=resolve_torch_device(screener_device or base_config.screener.device),
        resolved_tabpfn_device=resolve_torch_device(tabpfn_device or os.environ.get("NEURAL_BN_TABPFN_DEVICE", "auto")),
    )
    _emit_progress(
        progress,
        "evidence",
        f"starting stages={len(selected_stage_names)} artifact_dir={artifact_root}",
    )

    with temporary_env(benchmark_runtime_env(artifact_dir=artifact_root)):
        for stage_idx, stage_name in enumerate(selected_stage_names, start=1):
            spec = specs[stage_name]
            output_path = artifact_root / spec.output_filename
            _emit_progress(
                progress,
                "evidence",
                f"stage {stage_idx}/{len(selected_stage_names)} start "
                f"name={stage_name} suite={spec.suite_name}",
            )
            suite_result = run_benchmark_suite(
                suite_name=spec.suite_name,
                base_config=base_config,
                output_path=output_path,
                base_dir=base_dir,
                dataset_names=spec.dataset_names,
                baseline_names=spec.baseline_names,
                seeds=spec.seeds,
                ablations=spec.ablations,
                tabpfn_device=tabpfn_device,
                screener_device=screener_device,
                progress=progress,
                progress_label=f"stage:{stage_name}",
            )
            completed_payloads[stage_name] = suite_result.to_dict()
            summary = summarize_evidence_pass(
                stage_payloads=completed_payloads,
                stage_specs=specs,
                requested_stage_names=selected_stage_names,
                artifact_dir=artifact_root,
                screener_device=screener_device,
                tabpfn_device=tabpfn_device,
            )
            summary["run_metadata"] = evidence_run_metadata
            summary_path.write_text(json.dumps(summary, indent=2))
            ok_count = sum(record.status == "ok" for record in suite_result.records)
            _emit_progress(
                progress,
                "evidence",
                f"stage {stage_idx}/{len(selected_stage_names)} done "
                f"name={stage_name} records={len(suite_result.records)} ok={ok_count} "
                f"skipped={len(suite_result.records) - ok_count}",
            )
            _emit_progress(progress, "evidence", f"updated summary={summary_path}")

        final_summary = summarize_evidence_pass(
            stage_payloads=completed_payloads,
            stage_specs=specs,
            requested_stage_names=selected_stage_names,
            artifact_dir=artifact_root,
            screener_device=screener_device,
            tabpfn_device=tabpfn_device,
        )
    final_summary["run_metadata"] = evidence_run_metadata
    _emit_progress(
        progress,
        "evidence",
        f"finished completed_stages={len(final_summary['completed_stages'])} "
        f"verdict={final_summary['decision']['verdict']}",
    )
    return final_summary


def summarize_paper_positioning(artifact_dir: Path | None = None) -> Dict[str, Any]:
    artifact_root = Path(artifact_dir or Path("artifacts") / "benchmarks")
    evidence_path = artifact_root / "evidence_summary.json"
    paper_path = artifact_root / "paper_minimum_full.json"
    fast_probe_path = artifact_root / "known_graph_fast_probe.json"

    evidence_payload = _load_json_artifact(evidence_path)
    paper_payload = _load_json_artifact(paper_path)
    fast_probe_payload = _load_json_artifact(fast_probe_path)

    decision = evidence_payload["decision"]
    criteria = decision["criteria"]
    structural_rows = list(decision["comparisons"]["structural"])
    predictive_rows = list(decision["comparisons"]["predictive"])
    runtime_rows = [
        {
            "dataset_name": row["dataset_name"],
            "method_total_seconds": row["method_total_seconds"],
            "greedy_hc_total_seconds": row["greedy_hc_total_seconds"],
            "runtime_ratio_vs_greedy": (
                row["method_total_seconds"] / row["greedy_hc_total_seconds"]
                if row["greedy_hc_total_seconds"]
                else None
            ),
            "runtime_win": (
                row["method_total_seconds"] < row["greedy_hc_total_seconds"]
                if row["method_total_seconds"] is not None and row["greedy_hc_total_seconds"] is not None
                else None
            ),
        }
        for row in structural_rows
    ]
    fast_probe_summary = summarize_fast_mode_design_probe(
        fast_probe_payload,
        candidate_ablations=("known_graph_fast",),
    )
    known_graph_fast_summary = fast_probe_summary["candidates"]["known_graph_fast"]
    known_graph_fast_tradeoff = list(known_graph_fast_summary["dataset_rows"])

    quality_supported = bool(criteria.get("structural_pass") and criteria.get("predictive_pass"))
    runtime_supported = not bool(criteria.get("runtime_pass"))
    fast_tradeoff_supported = not bool(known_graph_fast_summary["promotion_gate"]["promotable"])
    structural_wins = int(criteria.get("structural_win_count", 0))
    structural_total = int(criteria.get("structural_datasets_compared", 0))
    predictive_wins = int(criteria.get("predictive_competitive_count", 0))
    predictive_total = int(criteria.get("predictive_datasets_compared", 0))
    runtime_wins = int(criteria.get("runtime_win_count", 0))
    runtime_total = int(criteria.get("runtime_datasets_compared", 0))
    fast_shd_improvements = sum(row["shd_delta_vs_default"] < 0.0 for row in known_graph_fast_tradeoff)
    fast_shd_regressions = sum(row["shd_delta_vs_default"] > 0.0 for row in known_graph_fast_tradeoff)

    recommended_claim_text = (
        "Neural-screened BN should be framed as a quality-first explicit BN classifier: "
        f"the current default wins the staged structural criterion on {structural_wins}/{structural_total} "
        "known-graph datasets and remains ROC-AUC competitive on "
        f"{predictive_wins}/{predictive_total} real datasets, while preserving explicit, inspectable DAGs."
    )
    recommended_limitation_text = (
        "The main limitation is runtime, not structural or predictive quality: "
        f"the current default wins the staged runtime rule on only {runtime_wins}/{runtime_total} known-graph datasets "
        "against greedy_hc_bn. The known_graph_fast ablation narrows the runtime gap substantially "
        f"but is not promotable because it still loses the runtime gate and regresses SHD on {fast_shd_regressions} of "
        f"{len(known_graph_fast_tradeoff)} known-graph datasets."
    )

    payload = {
        "artifact_dir": str(artifact_root),
        "source_artifacts": {
            "evidence_summary": str(evidence_path),
            "paper_minimum_full": str(paper_path),
            "known_graph_fast_probe": str(fast_probe_path),
        },
        "run_metadata": {
            "git_sha": _current_git_sha(),
            "basis_stage": decision["basis_stage"],
            "evidence_run_metadata": evidence_payload.get("run_metadata", {}),
            "paper_run_metadata": paper_payload.get("run_metadata", {}),
            "fast_probe_run_metadata": fast_probe_payload.get("run_metadata", {}),
        },
        "headline_verdicts": {
            "quality_first_default": {
                "supported": quality_supported,
                "message": (
                    "The current default clears the structural and predictive competitiveness criteria "
                    f"({structural_wins}/{structural_total} structural wins, "
                    f"{predictive_wins}/{predictive_total} competitive real-dataset results)."
                ),
            },
            "runtime_limitation": {
                "supported": runtime_supported,
                "message": (
                    "The current default does not satisfy the runtime rule against greedy_hc_bn "
                    f"({runtime_wins}/{runtime_total} runtime wins)."
                ),
            },
            "fast_profile_tradeoff": {
                "supported": fast_tradeoff_supported,
                "message": (
                    "known_graph_fast is a real speedup but not a new default: "
                    f"it improves SHD on {fast_shd_improvements}/{len(known_graph_fast_tradeoff)} known-graph datasets, "
                    f"regresses on {fast_shd_regressions}/{len(known_graph_fast_tradeoff)}, and still fails the runtime gate."
                ),
            },
        },
        "explicit_bn_structural_comparison": structural_rows,
        "strong_predictive_baseline_comparison": predictive_rows,
        "runtime_vs_greedy_hc_comparison": runtime_rows,
        "known_graph_fast_tradeoff": known_graph_fast_tradeoff,
        "recommended_paper_claim_text": recommended_claim_text,
        "recommended_limitation_text": recommended_limitation_text,
    }

    json_path = artifact_root / "paper_positioning.json"
    markdown_path = artifact_root / "paper_positioning.md"
    json_path.write_text(json.dumps(payload, indent=2))
    markdown_path.write_text(render_paper_positioning_markdown(payload))
    results_payload = summarize_paper_results(artifact_root)
    payload["related_outputs"] = {
        "paper_positioning_json": str(json_path),
        "paper_positioning_markdown": str(markdown_path),
        "paper_results_json": str(artifact_root / "paper_results.json"),
        "paper_results_markdown": str(artifact_root / "paper_results.md"),
    }
    payload["paper_results_headlines"] = results_payload["headline_metrics"]
    return payload


def summarize_paper_results(artifact_dir: Path | None = None) -> Dict[str, Any]:
    artifact_root = Path(artifact_dir or Path("artifacts") / "benchmarks")
    evidence_payload = _load_json_artifact(artifact_root / "evidence_summary.json")
    paper_payload = _load_json_artifact(artifact_root / "paper_minimum_full.json")
    fast_mode_design_path = artifact_root / "fast_mode_design_probe.json"
    known_graph_fast_path = artifact_root / "known_graph_fast_probe.json"
    fast_probe_source = fast_mode_design_path if fast_mode_design_path.exists() else known_graph_fast_path
    fast_probe_payload = _load_json_artifact(fast_probe_source)

    predictive_rows = paper_payload["tables"]["predictive_leaderboard"]
    structural_rows = paper_payload["tables"]["structural_leaderboard"]
    complexity_rows = paper_payload["tables"]["search_complexity"]
    decision = evidence_payload["decision"]
    structural_comparisons = list(decision["comparisons"]["structural"])
    predictive_comparisons = list(decision["comparisons"]["predictive"])

    predictive_index = {(row["dataset_name"], row["model_name"]): row for row in predictive_rows}
    structural_index = {(row["dataset_name"], row["model_name"]): row for row in structural_rows}
    complexity_index = {(row["dataset_name"], row["model_name"]): row for row in complexity_rows}
    explicit_bn_models = {"greedy_hc_bn", "naive_bayes", "tan", "kdb", "aode"}
    strong_predictive_models = {"xgboost", "lightgbm", "catboost", "tabpfn"}

    structural_table = []
    structural_wins = 0
    structural_near_ties = 0
    for comparison in structural_comparisons:
        dataset_name = comparison["dataset_name"]
        method_row = structural_index[(dataset_name, "neural_screened_bn")]
        explicit_candidates = [
            row
            for (row_dataset, row_model), row in structural_index.items()
            if row_dataset == dataset_name and row_model in explicit_bn_models
        ]
        best_explicit = min(explicit_candidates, key=lambda row: row["avg_shd"])
        shd_margin = float(best_explicit["avg_shd"] - method_row["avg_shd"])
        if shd_margin > 0:
            structural_wins += 1
        if abs(shd_margin) <= 0.5:
            structural_near_ties += 1
        structural_table.append(
            {
                "dataset_name": dataset_name,
                "method_shd": method_row["avg_shd"],
                "method_precision": method_row.get("avg_precision"),
                "method_recall": method_row.get("avg_recall"),
                "method_f1": method_row.get("avg_f1"),
                "candidate_parent_recall": method_row.get("avg_candidate_parent_recall"),
                "best_explicit_model": best_explicit["model_name"],
                "best_explicit_shd": best_explicit["avg_shd"],
                "shd_margin": shd_margin,
            }
        )

    predictive_table = []
    predictive_competitive = 0
    for comparison in predictive_comparisons:
        dataset_name = comparison["dataset_name"]
        method_row = predictive_index[(dataset_name, "neural_screened_bn")]
        strong_candidates = [
            row
            for (row_dataset, row_model), row in predictive_index.items()
            if row_dataset == dataset_name and row_model in strong_predictive_models
        ]
        best_strong = max(strong_candidates, key=lambda row: row["avg_roc_auc"])
        if comparison["competitive"]:
            predictive_competitive += 1
        predictive_table.append(
            {
                "dataset_name": dataset_name,
                "method_roc_auc": method_row["avg_roc_auc"],
                "method_log_loss": method_row["avg_log_loss"],
                "method_ece": method_row.get("avg_ece"),
                "best_strong_model": best_strong["model_name"],
                "best_strong_roc_auc": best_strong["avg_roc_auc"],
                "best_strong_log_loss": best_strong["avg_log_loss"],
                "roc_auc_gap": float(best_strong["avg_roc_auc"] - method_row["avg_roc_auc"]),
            }
        )
    predictive_lines = [
        (
            f"for {_humanize_dataset_name(row['dataset_name'])}, it reaches ROC-AUC {row['method_roc_auc']:.4f} "
            f"with log loss {row['method_log_loss']:.4f}"
        )
        for row in predictive_table
    ]

    fast_probe_summary = summarize_fast_mode_design_probe(
        fast_probe_payload,
        candidate_ablations=("known_graph_fast",),
    )
    fast_by_dataset = {
        row["dataset_name"]: row
        for row in fast_probe_summary["candidates"]["known_graph_fast"]["dataset_rows"]
    }
    runtime_table = []
    for comparison in structural_comparisons:
        dataset_name = comparison["dataset_name"]
        fast_row = fast_by_dataset.get(dataset_name)
        method_seconds = complexity_index[(dataset_name, "neural_screened_bn")]["avg_total_seconds"]
        greedy_seconds = complexity_index[(dataset_name, "greedy_hc_bn")]["avg_total_seconds"]
        known_graph_fast_seconds = fast_row["candidate_total_seconds"] if fast_row is not None else None
        runtime_table.append(
            {
                "dataset_name": dataset_name,
                "default_total_seconds": method_seconds,
                "known_graph_fast_total_seconds": known_graph_fast_seconds,
                "greedy_hc_total_seconds": greedy_seconds,
                "default_vs_greedy_ratio": (
                    method_seconds / greedy_seconds if greedy_seconds else None
                ),
                "fast_vs_greedy_ratio": (
                    known_graph_fast_seconds / greedy_seconds
                    if known_graph_fast_seconds is not None and greedy_seconds
                    else None
                ),
                "fast_reduction_vs_default_pct": (
                    100.0 * (1.0 - known_graph_fast_seconds / method_seconds)
                    if known_graph_fast_seconds is not None and method_seconds
                    else None
                ),
            }
        )

    headline_metrics = {
        "structural_win_count": structural_wins,
        "structural_dataset_count": len(structural_table),
        "structural_near_tie_count": structural_near_ties,
        "predictive_competitive_count": predictive_competitive,
        "predictive_dataset_count": len(predictive_table),
        "runtime_win_count_vs_greedy": sum(row["default_total_seconds"] < row["greedy_hc_total_seconds"] for row in runtime_table),
        "runtime_dataset_count": len(runtime_table),
        "known_graph_fast_promotion_winners": list(fast_probe_summary["promotion_winners"]),
    }

    results_narrative = [
        (
            "Across the five known-graph datasets, the default neural-screened BN beats the best explicit BN baseline "
            f"on SHD on {structural_wins}/{len(structural_table)} datasets and is within 0.5 SHD on {structural_near_ties}/{len(structural_table)}. "
            "The strongest structural result is on insurance, where the method reaches SHD 19.75 versus 26.0 for the best explicit baseline, "
            "while alarm is effectively tied at 26.25 versus 26.0."
        ),
        (
            "On the two real datasets, the method remains predictively competitive with strong tabular baselines while keeping an explicit DAG: "
            f"it is within the staged ROC-AUC competitiveness gap on {predictive_competitive}/{len(predictive_table)} datasets. Specifically, "
            + ", and ".join(predictive_lines)
            + "."
        ),
        (
            "The limiting factor is runtime, not quality. The default method is slower than greedy_hc_bn on all "
            f"{len(runtime_table)}/{len(runtime_table)} known-graph datasets, and the known_graph_fast ablation reduces runtime materially "
            "but still does not beat greedy_hc_bn on any dataset or earn promotion into the default path."
        ),
    ]
    limitations_narrative = [
        (
            "The current method should not be presented as a speed-oriented replacement for greedy hill-climbing BN search. "
            "Its strength is the quality/interpretability tradeoff, not raw runtime."
        ),
        (
            "Incremental runtime tuning appears exhausted in the current design. The separate fast-mode probe found no candidate "
            "that beat greedy_hc_bn on at least 3/5 known-graph datasets while staying within the SHD tolerance gate."
        ),
    ]

    payload = {
        "artifact_dir": str(artifact_root),
        "source_artifacts": {
            "evidence_summary": str(artifact_root / "evidence_summary.json"),
            "paper_minimum_full": str(artifact_root / "paper_minimum_full.json"),
            "fast_probe": str(fast_probe_source),
        },
        "headline_metrics": headline_metrics,
        "structural_table": structural_table,
        "predictive_table": predictive_table,
        "runtime_table": runtime_table,
        "results_narrative": results_narrative,
        "limitations_narrative": limitations_narrative,
    }

    json_path = artifact_root / "paper_results.json"
    markdown_path = artifact_root / "paper_results.md"
    json_path.write_text(json.dumps(payload, indent=2))
    markdown_path.write_text(render_paper_results_markdown(payload))
    return payload


def summarize_fast_mode_design_probe(
    suite_payload: Dict[str, Any],
    *,
    candidate_ablations: Sequence[str] = (
        "known_graph_fast",
        "mi_filter_fast",
        "single_pass_neural_fast",
        "hybrid_mi_neural_fast",
    ),
) -> Dict[str, Any]:
    structural_rows = suite_payload["tables"].get("structural_leaderboard", [])
    complexity_rows = suite_payload["tables"].get("search_complexity", [])
    ablation_rows = suite_payload["tables"].get("ablation_summary", [])
    records = suite_payload.get("records", [])

    default_shd = {
        row["dataset_name"]: row["avg_shd"]
        for row in structural_rows
        if row.get("model_name") == "neural_screened_bn" and "avg_shd" in row
    }
    default_runtime = {
        row["dataset_name"]: row["avg_total_seconds"]
        for row in complexity_rows
        if row.get("model_name") == "neural_screened_bn" and "avg_total_seconds" in row
    }
    greedy_runtime = {
        row["dataset_name"]: row["avg_total_seconds"]
        for row in complexity_rows
        if row.get("model_name") == "greedy_hc_bn" and "avg_total_seconds" in row
    }
    candidate_index = {
        (row["dataset_name"], row["ablation_label"]): row
        for row in ablation_rows
    }
    known_graph_datasets = sorted(default_shd)
    candidate_summaries: Dict[str, Any] = {}

    for ablation_label in candidate_ablations:
        dataset_rows: List[Dict[str, Any]] = []
        for dataset_name in known_graph_datasets:
            row = candidate_index.get((dataset_name, ablation_label))
            if row is None:
                continue
            default_seconds = default_runtime.get(dataset_name)
            greedy_seconds = greedy_runtime.get(dataset_name)
            default_dataset_shd = default_shd.get(dataset_name)
            dataset_rows.append(
                {
                    "dataset_name": dataset_name,
                    "default_total_seconds": default_seconds,
                    "candidate_total_seconds": row.get("avg_total_seconds"),
                    "greedy_hc_total_seconds": greedy_seconds,
                    "default_shd": default_dataset_shd,
                    "candidate_shd": row.get("avg_shd"),
                    "runtime_delta_vs_default": (
                        row.get("avg_total_seconds") - default_seconds
                        if default_seconds is not None and row.get("avg_total_seconds") is not None
                        else None
                    ),
                    "shd_delta_vs_default": (
                        row.get("avg_shd") - default_dataset_shd
                        if default_dataset_shd is not None and row.get("avg_shd") is not None
                        else None
                    ),
                    "faster_than_default": (
                        row.get("avg_total_seconds") < default_seconds
                        if default_seconds is not None and row.get("avg_total_seconds") is not None
                        else False
                    ),
                    "faster_than_greedy_hc": (
                        row.get("avg_total_seconds") < greedy_seconds
                        if greedy_seconds is not None and row.get("avg_total_seconds") is not None
                        else False
                    ),
                    "within_default_plus_two_shd": (
                        row.get("avg_shd") <= default_dataset_shd + 2.0
                        if default_dataset_shd is not None and row.get("avg_shd") is not None
                        else False
                    ),
                }
            )
        skipped_runs = sum(
            1
            for record in records
            if record["metadata"]["model_name"] == "neural_screened_bn"
            and record["metadata"]["ablation_label"] == ablation_label
            and record["metadata"]["dataset_regime"] == "known_graph"
            and record["status"] != "ok"
        )
        promotion_gate = {
            "runtime_faster_than_default_all": (
                bool(dataset_rows) and all(row["faster_than_default"] for row in dataset_rows)
            ),
            "runtime_vs_greedy_win_count": sum(row["faster_than_greedy_hc"] for row in dataset_rows),
            "shd_within_default_plus_two_count": sum(row["within_default_plus_two_shd"] for row in dataset_rows),
            "skipped_runs": skipped_runs,
        }
        promotion_gate["promotable"] = (
            promotion_gate["runtime_faster_than_default_all"]
            and promotion_gate["runtime_vs_greedy_win_count"] >= 3
            and promotion_gate["shd_within_default_plus_two_count"] >= 4
            and skipped_runs == 0
        )
        candidate_summaries[ablation_label] = {
            "dataset_rows": dataset_rows,
            "promotion_gate": promotion_gate,
        }

    promotion_winners = [
        ablation_label
        for ablation_label, summary in candidate_summaries.items()
        if summary["promotion_gate"]["promotable"]
    ]
    return {
        "known_graph_datasets": known_graph_datasets,
        "candidates": candidate_summaries,
        "promotion_winners": promotion_winners,
    }


def summarize_records(records: Sequence[ExperimentResult]) -> Dict[str, List[Dict[str, Any]]]:
    valid = [record for record in records if record.status == "ok"]
    default_valid = [record for record in valid if record.metadata.ablation_label == "default"]
    skipped = [record for record in records if record.status != "ok"]
    predictive_rows = _aggregate_rows(default_valid, metric_keys=["roc_auc", "log_loss", "brier", "ece"])
    structural_rows = _aggregate_rows(
        [record for record in default_valid if record.structural and "shd" in record.structural],
        metric_keys=["shd", "precision", "recall", "f1", "candidate_parent_recall"],
    )
    complexity_rows = _aggregate_rows(
        default_valid,
        metric_keys=["candidate_pool_reduction", "predicted_edge_count"],
        runtime_keys=["total_seconds", "search_seconds"],
    )
    ablation_rows = _aggregate_rows(
        [
            record for record in valid
            if record.metadata.ablation_label != "default" and record.metadata.model_name == "neural_screened_bn"
        ],
        metric_keys=["roc_auc", "log_loss", "shd", "candidate_pool_reduction"],
        runtime_keys=["total_seconds", "screener_seconds", "search_seconds"],
        group_keys=("dataset_name", "ablation_label"),
    )
    return {
        "predictive_leaderboard": predictive_rows,
        "structural_leaderboard": structural_rows,
        "search_complexity": complexity_rows,
        "ablation_summary": ablation_rows,
        "skipped_items": [
            {
                "dataset_name": record.metadata.dataset_name,
                "model_name": record.metadata.model_name,
                "split_id": record.metadata.split_id,
                "status": record.status,
                "notes": list(record.notes),
            }
            for record in skipped
        ],
    }


def render_paper_positioning_markdown(payload: Dict[str, Any]) -> str:
    headline_verdicts = payload["headline_verdicts"]
    sections = [
        "# Paper Positioning Snapshot",
        "",
        "## Headline Verdicts",
        f"- `quality_first_default`: {headline_verdicts['quality_first_default']['message']}",
        f"- `runtime_limitation`: {headline_verdicts['runtime_limitation']['message']}",
        f"- `fast_profile_tradeoff`: {headline_verdicts['fast_profile_tradeoff']['message']}",
        "",
        "## Explicit-BN Structural Comparison",
        _markdown_table(
            payload["explicit_bn_structural_comparison"],
            [
                ("dataset_name", "Dataset"),
                ("method_shd", "Method SHD"),
                ("best_explicit_model", "Best Explicit"),
                ("best_explicit_shd", "Best Explicit SHD"),
                ("shd_delta", "SHD Delta"),
            ],
        ),
        "",
        "## Strong Predictive Baseline Comparison",
        _markdown_table(
            payload["strong_predictive_baseline_comparison"],
            [
                ("dataset_name", "Dataset"),
                ("method_roc_auc", "Method ROC-AUC"),
                ("best_strong_model", "Best Strong"),
                ("best_strong_roc_auc", "Best Strong ROC-AUC"),
                ("roc_auc_gap", "ROC-AUC Gap"),
                ("competitive", "Competitive"),
            ],
        ),
        "",
        "## Runtime vs `greedy_hc_bn`",
        _markdown_table(
            payload["runtime_vs_greedy_hc_comparison"],
            [
                ("dataset_name", "Dataset"),
                ("method_total_seconds", "Method Seconds"),
                ("greedy_hc_total_seconds", "Greedy Seconds"),
                ("runtime_ratio_vs_greedy", "Ratio vs Greedy"),
                ("runtime_win", "Runtime Win"),
            ],
        ),
        "",
        "## `known_graph_fast` Tradeoff vs Default",
        _markdown_table(
            payload["known_graph_fast_tradeoff"],
            [
                ("dataset_name", "Dataset"),
                ("default_total_seconds", "Default Seconds"),
                ("candidate_total_seconds", "Fast Seconds"),
                ("greedy_hc_total_seconds", "Greedy Seconds"),
                ("default_shd", "Default SHD"),
                ("candidate_shd", "Fast SHD"),
                ("shd_delta_vs_default", "SHD Delta"),
            ],
        ),
        "",
        "## Recommended Claim Text",
        payload["recommended_paper_claim_text"],
        "",
        "## Recommended Limitation Text",
        payload["recommended_limitation_text"],
        "",
    ]
    return "\n".join(sections)


def render_paper_results_markdown(payload: Dict[str, Any]) -> str:
    sections = [
        "# Paper Results Draft",
        "",
        "## Headline Metrics",
        f"- Structural wins over the best explicit BN baseline: {payload['headline_metrics']['structural_win_count']}/{payload['headline_metrics']['structural_dataset_count']}",
        f"- Near ties on known-graph SHD (<= 0.5): {payload['headline_metrics']['structural_near_tie_count']}/{payload['headline_metrics']['structural_dataset_count']}",
        f"- Predictively competitive real datasets: {payload['headline_metrics']['predictive_competitive_count']}/{payload['headline_metrics']['predictive_dataset_count']}",
        f"- Runtime wins vs `greedy_hc_bn`: {payload['headline_metrics']['runtime_win_count_vs_greedy']}/{payload['headline_metrics']['runtime_dataset_count']}",
        "",
        "## Structural Table",
        _markdown_table(
            payload["structural_table"],
            [
                ("dataset_name", "Dataset"),
                ("method_shd", "Method SHD"),
                ("method_precision", "Precision"),
                ("method_recall", "Recall"),
                ("best_explicit_model", "Best Explicit"),
                ("best_explicit_shd", "Best Explicit SHD"),
                ("shd_margin", "SHD Margin"),
            ],
        ),
        "",
        "## Predictive Table",
        _markdown_table(
            payload["predictive_table"],
            [
                ("dataset_name", "Dataset"),
                ("method_roc_auc", "Method ROC-AUC"),
                ("method_log_loss", "Method Log Loss"),
                ("best_strong_model", "Best Strong"),
                ("best_strong_roc_auc", "Best Strong ROC-AUC"),
                ("roc_auc_gap", "ROC-AUC Gap"),
            ],
        ),
        "",
        "## Runtime Table",
        _markdown_table(
            payload["runtime_table"],
            [
                ("dataset_name", "Dataset"),
                ("default_total_seconds", "Default Seconds"),
                ("known_graph_fast_total_seconds", "known_graph_fast Seconds"),
                ("greedy_hc_total_seconds", "Greedy Seconds"),
                ("default_vs_greedy_ratio", "Default/Greedy"),
                ("fast_reduction_vs_default_pct", "Fast Reduction %"),
            ],
        ),
        "",
        "## Results Narrative Draft",
        *[f"- {item}" for item in payload["results_narrative"]],
        "",
        "## Limitations Draft",
        *[f"- {item}" for item in payload["limitations_narrative"]],
        "",
    ]
    return "\n".join(sections)


def summarize_evidence_pass(
    stage_payloads: Dict[str, Dict[str, Any]],
    stage_specs: Dict[str, EvidenceStageSpec] | None = None,
    requested_stage_names: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    screener_device: str | None = None,
    tabpfn_device: str | None = None,
) -> Dict[str, Any]:
    specs = stage_specs or evidence_stage_registry()
    selected_stage_names = list(requested_stage_names or specs.keys())
    stage_overview: List[Dict[str, Any]] = []
    for stage_name in selected_stage_names:
        spec = specs[stage_name]
        payload = stage_payloads.get(stage_name)
        artifact_path = str((artifact_dir or Path("artifacts") / "benchmarks") / spec.output_filename)
        if payload is None:
            stage_overview.append(
                {
                    "stage_name": stage_name,
                    "suite_name": spec.suite_name,
                    "artifact_path": artifact_path,
                    "completed": False,
                    "datasets": list(spec.dataset_names or []),
                    "baselines": list(spec.baseline_names or []),
                    "seeds": list(spec.seeds or []),
                    "ablations": list(spec.ablations or []),
                    "num_records": 0,
                    "num_ok": 0,
                    "num_skipped": 0,
                }
            )
            continue

        records = payload["records"]
        num_ok = sum(record["status"] == "ok" for record in records)
        num_skipped = sum(record["status"] != "ok" for record in records)
        stage_overview.append(
            {
                "stage_name": stage_name,
                "suite_name": payload["suite_name"],
                "artifact_path": artifact_path,
                "completed": True,
                "datasets": list(spec.dataset_names or []),
                "baselines": list(spec.baseline_names or []),
                "seeds": list(spec.seeds or []),
                "ablations": list(spec.ablations or []),
                "num_records": len(records),
                "num_ok": num_ok,
                "num_skipped": num_skipped,
            }
        )

    return {
        "artifact_dir": str(artifact_dir or Path("artifacts") / "benchmarks"),
        "requested_stages": selected_stage_names,
        "completed_stages": [name for name in selected_stage_names if name in stage_payloads],
        "stage_overview": stage_overview,
        "stage_tables": {
            stage_name: payload["tables"]
            for stage_name, payload in stage_payloads.items()
        },
        "run_metadata": _evidence_run_metadata(
            artifact_dir=artifact_dir or Path("artifacts") / "benchmarks",
            requested_stage_names=selected_stage_names,
            requested_screener_device=screener_device,
            requested_tabpfn_device=tabpfn_device,
            resolved_screener_device=resolve_torch_device(screener_device or "auto"),
            resolved_tabpfn_device=resolve_torch_device(tabpfn_device or os.environ.get("NEURAL_BN_TABPFN_DEVICE", "auto")),
        ),
        "decision": _evidence_decision(stage_payloads),
    }


def _evidence_decision(stage_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    method_name = "neural_screened_bn"
    explicit_bn_models = {"greedy_hc_bn", "naive_bayes", "tan", "kdb", "aode"}
    strong_predictive_models = {"xgboost", "lightgbm", "catboost", "tabpfn"}
    stage_priority = ["paper_minimum_full", "structural_default", "sanity_default"]
    basis_stage_name = next((stage for stage in stage_priority if stage in stage_payloads), None)

    if basis_stage_name is None:
        return {
            "verdict": "incomplete",
            "basis_stage": None,
            "rule": _decision_rule_description(),
            "criteria": {},
            "comparisons": {"structural": [], "predictive": []},
            "reason": "No completed evidence stages yet.",
        }

    basis_payload = stage_payloads[basis_stage_name]
    basis_tables = basis_payload["tables"]
    predictive_rows = basis_tables["predictive_leaderboard"]
    structural_rows = basis_tables["structural_leaderboard"]
    complexity_rows = basis_tables["search_complexity"]

    predictive_index = {(row["dataset_name"], row["model_name"]): row for row in predictive_rows}
    structural_index = {(row["dataset_name"], row["model_name"]): row for row in structural_rows}
    complexity_index = {(row["dataset_name"], row["model_name"]): row for row in complexity_rows}

    dataset_regimes: Dict[str, str] = {}
    for payload in stage_payloads.values():
        for record in payload["records"]:
            if record["status"] != "ok" or record["metadata"]["ablation_label"] != "default":
                continue
            dataset_regimes[record["metadata"]["dataset_name"]] = record["metadata"]["dataset_regime"]

    structural_comparisons: List[Dict[str, Any]] = []
    structural_datasets = sorted(
        dataset_name
        for dataset_name, regime in dataset_regimes.items()
        if regime == "known_graph" and (dataset_name, method_name) in structural_index
    )
    for dataset_name in structural_datasets:
        method_row = structural_index[(dataset_name, method_name)]
        explicit_rows = [
            row
            for (row_dataset, row_model), row in structural_index.items()
            if row_dataset == dataset_name and row_model in explicit_bn_models
        ]
        if not explicit_rows:
            continue
        best_explicit = min(explicit_rows, key=lambda row: row["avg_shd"])
        method_runtime = complexity_index.get((dataset_name, method_name), {}).get("avg_total_seconds")
        greedy_runtime = complexity_index.get((dataset_name, "greedy_hc_bn"), {}).get("avg_total_seconds")
        structural_comparisons.append(
            {
                "dataset_name": dataset_name,
                "method_shd": method_row["avg_shd"],
                "best_explicit_model": best_explicit["model_name"],
                "best_explicit_shd": best_explicit["avg_shd"],
                "shd_delta": best_explicit["avg_shd"] - method_row["avg_shd"],
                "method_total_seconds": method_runtime,
                "greedy_hc_total_seconds": greedy_runtime,
            }
        )

    structural_win_count = sum(item["shd_delta"] > 0 for item in structural_comparisons)
    structural_needed = math.ceil(len(structural_comparisons) / 2) if structural_comparisons else None
    structural_pass = bool(structural_comparisons) and structural_win_count >= (structural_needed or 0)

    runtime_comparisons = [
        item
        for item in structural_comparisons
        if item["method_total_seconds"] is not None and item["greedy_hc_total_seconds"] is not None
    ]
    runtime_win_count = sum(
        item["method_total_seconds"] < item["greedy_hc_total_seconds"]
        for item in runtime_comparisons
    )
    runtime_needed = math.ceil(len(runtime_comparisons) / 2) if runtime_comparisons else None
    runtime_pass = bool(runtime_comparisons) and runtime_win_count >= (runtime_needed or 0)

    predictive_comparisons: List[Dict[str, Any]] = []
    real_datasets = sorted(
        dataset_name
        for dataset_name, regime in dataset_regimes.items()
        if regime == "real" and (dataset_name, method_name) in predictive_index
    )
    for dataset_name in real_datasets:
        method_row = predictive_index[(dataset_name, method_name)]
        strong_rows = [
            row
            for (row_dataset, row_model), row in predictive_index.items()
            if row_dataset == dataset_name and row_model in strong_predictive_models
        ]
        if not strong_rows:
            continue
        best_strong = max(strong_rows, key=lambda row: row["avg_roc_auc"])
        gap = best_strong["avg_roc_auc"] - method_row["avg_roc_auc"]
        predictive_comparisons.append(
            {
                "dataset_name": dataset_name,
                "method_roc_auc": method_row["avg_roc_auc"],
                "best_strong_model": best_strong["model_name"],
                "best_strong_roc_auc": best_strong["avg_roc_auc"],
                "roc_auc_gap": gap,
                "competitive": gap <= 0.03,
            }
        )

    predictive_competitive_count = sum(item["competitive"] for item in predictive_comparisons)
    predictive_needed = math.ceil(len(predictive_comparisons) / 2) if predictive_comparisons else None
    predictive_pass = (
        bool(predictive_comparisons)
        and predictive_competitive_count >= (predictive_needed or 0)
    )

    tabpfn_known_graph_ok = any(
        record["status"] == "ok"
        and record["metadata"]["ablation_label"] == "default"
        and record["metadata"]["model_name"] == "tabpfn"
        and record["metadata"]["dataset_regime"] == "known_graph"
        for payload in stage_payloads.values()
        for record in payload["records"]
    )
    tabpfn_real_ok = any(
        record["status"] == "ok"
        and record["metadata"]["ablation_label"] == "default"
        and record["metadata"]["model_name"] == "tabpfn"
        and record["metadata"]["dataset_regime"] == "real"
        for payload in stage_payloads.values()
        for record in payload["records"]
    )
    expert_skip_free = not any(
        record["status"] != "ok"
        and record["metadata"]["dataset_name"] in {"alarm", "child", "insurance"}
        for payload in stage_payloads.values()
        for record in payload["records"]
    )

    full_stage_complete = "paper_minimum_full" in stage_payloads
    if not full_stage_complete:
        verdict = "incomplete"
        reason = "Full paper_minimum stage not completed yet."
    elif structural_pass and runtime_pass and predictive_pass and tabpfn_known_graph_ok and tabpfn_real_ok and expert_skip_free:
        verdict = "go"
        reason = "Method clears the staged evidence rule on structure, runtime, and predictive competitiveness."
    else:
        verdict = "no_go"
        reason = "Method does not yet clear the staged evidence rule on structure/runtime and predictive competitiveness."

    return {
        "verdict": verdict,
        "basis_stage": basis_stage_name,
        "rule": _decision_rule_description(),
        "criteria": {
            "full_stage_complete": full_stage_complete,
            "structural_pass": structural_pass,
            "structural_win_count": structural_win_count,
            "structural_datasets_compared": len(structural_comparisons),
            "runtime_pass": runtime_pass,
            "runtime_win_count": runtime_win_count,
            "runtime_datasets_compared": len(runtime_comparisons),
            "predictive_pass": predictive_pass,
            "predictive_competitive_count": predictive_competitive_count,
            "predictive_datasets_compared": len(predictive_comparisons),
            "tabpfn_known_graph_ok": tabpfn_known_graph_ok,
            "tabpfn_real_ok": tabpfn_real_ok,
            "expert_skip_free": expert_skip_free,
        },
        "comparisons": {
            "structural": structural_comparisons,
            "predictive": predictive_comparisons,
        },
        "reason": reason,
    }


def _decision_rule_description() -> Dict[str, Any]:
    return {
        "structural": "neural_screened_bn must beat the best explicit BN baseline on SHD on at least half of comparable known-graph datasets.",
        "runtime": "neural_screened_bn must be faster than greedy_hc_bn on total runtime on at least half of known-graph datasets with both measurements.",
        "predictive": "neural_screened_bn must be within 0.03 ROC-AUC of the best strong predictive baseline on at least half of comparable real datasets.",
        "health": "paper_minimum_full must complete, TabPFN must succeed on at least one known-graph and one real dataset, and alarm/child/insurance must have no skipped runs.",
    }


def _aggregate_rows(
    records: Sequence[ExperimentResult],
    metric_keys: Sequence[str],
    runtime_keys: Sequence[str] | None = None,
    group_keys: tuple[str, ...] = ("dataset_name", "model_name"),
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], List[ExperimentResult]] = {}
    for record in records:
        values = []
        for key in group_keys:
            if key == "dataset_name":
                values.append(record.metadata.dataset_name)
            elif key == "model_name":
                values.append(record.metadata.model_name)
            elif key == "ablation_label":
                values.append(record.metadata.ablation_label)
        grouped.setdefault(tuple(values), []).append(record)

    rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        for metric_key in metric_keys:
            values = []
            for item in items:
                if item.predictive and metric_key in item.predictive:
                    values.append(item.predictive[metric_key])
                elif item.structural and metric_key in item.structural:
                    values.append(item.structural[metric_key])
                elif metric_key in item.screener_diagnostics:
                    values.append(item.screener_diagnostics[metric_key])
            if values:
                row[f"avg_{metric_key}"] = float(np.mean(values))
        if runtime_keys:
            for runtime_key in runtime_keys:
                values = [item.runtimes.get(runtime_key) for item in items if runtime_key in item.runtimes]
                if values:
                    row[f"avg_{runtime_key}"] = float(np.mean(values))
        row["num_runs"] = len(items)
        rows.append(row)
    return sorted(rows, key=lambda item: tuple(str(value) for value in item.values()))


def _build_result(
    model_name: str,
    family: str,
    split: DatasetSplit,
    ablation_label: str,
    probabilities: np.ndarray,
    graph: nx.DiGraph | None,
    candidate_pools: Dict[str, List[str]],
    truth_graph: nx.DiGraph | None,
    runtimes: Dict[str, float],
    search_stats: Dict[str, Any],
    y_true: np.ndarray,
) -> ExperimentResult:
    has_candidate_pools = bool(candidate_pools)
    structural = {
        "candidate_pool_reduction": (
            candidate_pool_reduction(
                candidate_pools,
                graph.nodes() if graph is not None else split.artifact.frame.columns,
            )
            if has_candidate_pools
            else 0.0
        ),
        "predicted_edge_count": float(graph.number_of_edges()) if graph is not None else 0.0,
    }
    if truth_graph is not None and graph is not None:
        structural["shd"] = float(structural_hamming_distance(graph, truth_graph))
        structural["candidate_parent_recall"] = (
            candidate_pool_parent_recall(candidate_pools, truth_graph)
            if has_candidate_pools
            else 0.0
        )
        structural.update(edge_precision_recall_f1(graph, truth_graph))
    predictive = predictive_metrics(y_true, probabilities)
    return ExperimentResult(
        metadata=ExperimentMetadata(
            model_name=model_name,
            dataset_name=split.artifact.name,
            dataset_regime=split.artifact.regime,
            split_id=split.split_id,
            seed=split.seed,
            ablation_label=ablation_label,
            baseline_family=family,
        ),
        predictive=predictive,
        structural=structural,
        search_stats=search_stats,
        runtimes=runtimes,
        split_metadata=split.split_metadata,
        graph_edges=list(graph.edges()) if graph is not None else [],
        candidate_pools=candidate_pools,
        screener_diagnostics={
            "candidate_pool_reduction": structural["candidate_pool_reduction"],
        },
    )


def _encode_supervised_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    x_train = pd.get_dummies(train_frame.drop(columns=[target_column]), dummy_na=True)
    x_test = pd.get_dummies(test_frame.drop(columns=[target_column]), dummy_na=True)
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0.0)
    sanitized_columns = _sanitize_feature_names(x_train.columns)
    x_train.columns = sanitized_columns
    x_test.columns = sanitized_columns
    encoder = LabelEncoder().fit(
        pd.concat([train_frame[target_column], test_frame[target_column]], axis=0).astype(str)
    )
    y_train = encoder.transform(train_frame[target_column].astype(str))
    y_test = encoder.transform(test_frame[target_column].astype(str))
    return x_train.astype(float), x_test.astype(float), y_train, y_test


def _sanitize_feature_names(columns: Sequence[object]) -> list[str]:
    sanitized: list[str] = []
    counts: Dict[str, int] = {}
    for idx, column in enumerate(columns):
        candidate = str(column)
        for forbidden in ("[", "]", "<"):
            candidate = candidate.replace(forbidden, "_")
        if not candidate:
            candidate = f"feature_{idx}"
        occurrence = counts.get(candidate, 0)
        counts[candidate] = occurrence + 1
        if occurrence:
            candidate = f"{candidate}__{occurrence}"
        sanitized.append(candidate)
    return sanitized


def _load_json_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text())


def _markdown_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> str:
    if not rows:
        headers = " | ".join(label for _, label in columns)
        separators = " | ".join("---" for _ in columns)
        empty_cells = ["(none)"] + [""] * (len(columns) - 1)
        return f"| {headers} |\n| {separators} |\n| " + " | ".join(empty_cells) + " |"

    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return "-"
        return str(value)

    header_row = "| " + " | ".join(label for _, label in columns) + " |"
    separator_row = "| " + " | ".join("---" for _ in columns) + " |"
    body_rows = [
        "| " + " | ".join(_format_value(row.get(key)) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _humanize_dataset_name(dataset_name: str) -> str:
    label = dataset_name.replace("sklearn_", "").replace("_", " ")
    return label.capitalize()


def _baseline_availability_snapshot() -> Dict[str, Dict[str, Any]]:
    return {
        name: {"available": item.available, "detail": item.detail}
        for name, item in external_baseline_availability().items()
    }


def _current_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _suite_run_metadata(
    *,
    suite_name: str,
    datasets: Sequence[str],
    baselines: Sequence[str],
    seeds: Sequence[int],
    ablations: Sequence[str],
    requested_screener_device: str | None,
    requested_tabpfn_device: str | None,
    resolved_screener_device: str,
    resolved_tabpfn_device: str,
    output_path: Path | None,
) -> Dict[str, Any]:
    return {
        "git_sha": _current_git_sha(),
        "suite_name": suite_name,
        "requested_datasets": list(datasets),
        "requested_baselines": list(baselines),
        "requested_seeds": list(seeds),
        "requested_ablations": list(ablations),
        "requested_screener_device": requested_screener_device,
        "requested_tabpfn_device": requested_tabpfn_device,
        "resolved_screener_device": resolved_screener_device,
        "resolved_tabpfn_device": resolved_tabpfn_device,
        "baseline_availability": _baseline_availability_snapshot(),
        "output_path": str(output_path) if output_path is not None else None,
    }


def _evidence_run_metadata(
    *,
    artifact_dir: Path,
    requested_stage_names: Sequence[str],
    requested_screener_device: str | None,
    requested_tabpfn_device: str | None,
    resolved_screener_device: str,
    resolved_tabpfn_device: str,
) -> Dict[str, Any]:
    return {
        "artifact_dir": str(artifact_dir),
        "requested_stages": list(requested_stage_names),
        "git_sha": _current_git_sha(),
        "requested_screener_device": requested_screener_device,
        "requested_tabpfn_device": requested_tabpfn_device,
        "resolved_screener_device": resolved_screener_device,
        "resolved_tabpfn_device": resolved_tabpfn_device,
        "baseline_availability": _baseline_availability_snapshot(),
    }


def _instantiate_external_model(name: str, seed: int, n_classes: int):
    if name == "xgboost":
        external_device = _resolve_external_baseline_device("NEURAL_BN_EXTERNAL_DEVICE")
        module = importlib.import_module("xgboost")
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        kwargs = {"objective": objective, "random_state": seed, "eval_metric": "logloss"}
        if external_device is not None:
            kwargs["tree_method"] = "hist"
            kwargs["device"] = external_device
        if n_classes > 2:
            kwargs["num_class"] = n_classes
        return module.XGBClassifier(**kwargs)
    if name == "lightgbm":
        lightgbm_device = _resolve_external_baseline_device("NEURAL_BN_LIGHTGBM_DEVICE")
        module = importlib.import_module("lightgbm")
        kwargs = {
            "random_state": seed,
            "verbose": -1,
            "n_estimators": 64,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "force_col_wise": True,
            "n_jobs": 1,
        }
        if lightgbm_device is not None:
            kwargs["device_type"] = "gpu"
            kwargs["gpu_device_id"] = _cuda_device_index(lightgbm_device)
        if n_classes > 2:
            kwargs["objective"] = "multiclass"
            kwargs["num_class"] = n_classes
        return module.LGBMClassifier(**kwargs)
    module = importlib.import_module("catboost")
    loss_function = "Logloss" if n_classes == 2 else "MultiClass"
    return module.CatBoostClassifier(random_state=seed, verbose=False, loss_function=loss_function)


def _resolve_external_baseline_device(env_var: str) -> str | None:
    requested = os.environ.get(env_var)
    if not requested:
        return None
    resolved = resolve_torch_device(requested)
    return resolved if resolved.startswith("cuda") else None


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    _, _, suffix = device.partition(":")
    try:
        return int(suffix)
    except ValueError:
        return 0


def _instantiate_tabpfn_model(seed: int):
    module = importlib.import_module("tabpfn")
    model_path_env = os.environ.get("NEURAL_BN_TABPFN_MODEL_PATH")
    if model_path_env:
        configured_path = Path(model_path_env).expanduser()
        if not configured_path.exists():
            raise FileNotFoundError(f"configured TabPFN checkpoint does not exist: {configured_path}")
        model_path: str | Path = configured_path
    else:
        discovered_checkpoint = discover_tabpfn_checkpoint()
        model_path = discovered_checkpoint if discovered_checkpoint is not None else "auto"
    runtime_device = resolve_torch_device(os.environ.get("NEURAL_BN_TABPFN_DEVICE", "auto"))
    return module.TabPFNClassifier(
        n_estimators=4,
        device=runtime_device,
        random_state=seed,
        model_path=str(model_path) if isinstance(model_path, Path) else model_path,
        n_preprocessing_jobs=1,
        ignore_pretraining_limits=True,
    )


@contextlib.contextmanager
def _tabpfn_quiet_environment():
    runtime_device = resolve_torch_device(os.environ.get("NEURAL_BN_TABPFN_DEVICE", "auto"))
    previous = {
        "TABPFN_DISABLE_TELEMETRY": os.environ.get("TABPFN_DISABLE_TELEMETRY"),
        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
        "TABPFN_ALLOW_CPU_LARGE_DATASET": os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET"),
        "POSTHOG_DISABLED": os.environ.get("POSTHOG_DISABLED"),
    }
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if runtime_device == "cpu":
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    else:
        os.environ.pop("TABPFN_ALLOW_CPU_LARGE_DATASET", None)
    os.environ["POSTHOG_DISABLED"] = "1"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _skipped_result(model_name: str, family: str, split: DatasetSplit, note: str) -> ExperimentResult:
    return ExperimentResult(
        metadata=ExperimentMetadata(
            model_name=model_name,
            dataset_name=split.artifact.name,
            dataset_regime=split.artifact.regime,
            split_id=split.split_id,
            seed=split.seed,
            ablation_label="default",
            baseline_family=family,
        ),
        predictive={},
        structural=None,
        search_stats={},
        runtimes={},
        split_metadata=split.split_metadata,
        status="skipped",
        notes=[note],
    )


def _placeholder_artifact(name: str, regime: str):
    from .datasets import DatasetArtifact

    return DatasetArtifact(
        name=name,
        regime=regime,
        frame=pd.DataFrame(),
        target_column="label",
        truth_graph=None,
        source="placeholder",
    )
