"""End-to-end estimator and evaluation helpers."""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import TabularPreprocessor
from .metrics import (
    candidate_pool_parent_recall,
    candidate_pool_reduction,
    edge_precision_recall_f1,
    predictive_metrics,
    structural_hamming_distance,
)
from .results import ExperimentMetadata, ExperimentResult
from .models import DiscreteBayesianNetwork
from .screener import NeuralDependencyScreener, ScreenerResult
from .search import ConstrainedBNStructureLearner, SearchResult


class NeuralScreenedBNClassifier:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.preprocessor_ = TabularPreprocessor(
            target_column=config.target_column,
            n_bins=config.n_bins,
            min_samples_per_bin=config.min_samples_per_bin,
        )
        self.screener_ = NeuralDependencyScreener(config.screener)
        self.search_ = ConstrainedBNStructureLearner(config.search)
        self.model_: DiscreteBayesianNetwork | None = None
        self.graph_: nx.DiGraph | None = None
        self.result_: ExperimentResult | None = None
        self.target_categories_: list[str] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None = None,
        truth_graph: nx.DiGraph | None = None,
        dataset_name: str = "in_memory",
        dataset_regime: str = "unknown",
        split_id: str = "fit",
        seed: int | None = None,
        ablation_label: str = "default",
    ) -> "NeuralScreenedBNClassifier":
        t0 = perf_counter()
        self._log_progress(f"fit start dataset={dataset_name} split={split_id}")
        prepared = self.preprocessor_.fit_transform(frame)
        t1 = perf_counter()
        target_info = next(column for column in prepared.columns if column.name == self.config.target_column)
        self.target_categories_ = target_info.categories

        if self.config.use_screener:
            if self.config.screener.strategy == "mi_filter":
                screener_result = self.screener_.fit(prepared)
            elif self.config.screener.strategy == "hybrid_mi_neural":
                mi_allowed_sources = self.screener_.mutual_information_prefilter(
                    prepared,
                    limit=self.config.screener.mi_prefilter_size,
                )
                screener_result = self.screener_.fit(prepared, allowed_sources=mi_allowed_sources)
            else:
                screener_result = self.screener_.fit(prepared)
        else:
            screener_result = self._full_screener_result(prepared.discrete_frame.columns)
        t2 = perf_counter()
        validation_discrete = (
            self.preprocessor_.transform_for_prediction(validation_frame)
            if validation_frame is not None
            else None
        )
        candidate_support = self._search_candidate_support(screener_result)
        self._log_progress(
            f"search start columns={len(prepared.discrete_frame.columns)} "
            f"candidate_targets={len(screener_result.candidate_pools)}"
        )
        search_result = self.search_.fit(
            frame=prepared.discrete_frame,
            target_column=self.config.target_column,
            candidate_pools=screener_result.candidate_pools,
            candidate_scores=candidate_support,
            validation_frame=validation_discrete,
            dataset_regime=dataset_regime,
        )
        t3 = perf_counter()
        self._log_progress(
            f"search done edges={search_result.graph.number_of_edges()} "
            f"steps={len(search_result.operations)} runtime={t3 - t2:.1f}s"
        )
        self.graph_ = search_result.graph
        self._log_progress("final BN fit start")
        self.model_ = DiscreteBayesianNetwork(
            target_column=self.config.target_column,
            laplace=self.config.search.laplace,
        ).fit(prepared.discrete_frame, self.graph_)
        t4 = perf_counter()
        self._log_progress(f"final BN fit done total_runtime={t4 - t0:.1f}s")
        predictive = predictive_metrics(
            prepared.discrete_frame[self.config.target_column].to_numpy(dtype=int),
            self.model_.predict_proba(prepared.discrete_frame.drop(columns=[self.config.target_column])),
        )
        structural = self._structure_metrics(
            truth_graph=truth_graph,
            predicted=self.graph_,
            candidate_pools=screener_result.candidate_pools,
        )
        self.result_ = ExperimentResult(
            metadata=ExperimentMetadata(
                model_name="neural_screened_bn",
                dataset_name=dataset_name,
                dataset_regime=dataset_regime,
                split_id=split_id,
                seed=seed,
                ablation_label=ablation_label,
                baseline_family="method",
            ),
            predictive=predictive,
            structural=structural,
            search_stats={
                "total_score": search_result.total_score,
                "generative_score": search_result.generative_score,
                "discriminative_score": search_result.discriminative_score,
                "prior_score": search_result.prior_score,
                "hub_penalty": search_result.hub_penalty,
                "num_steps": len(search_result.operations),
                "num_edges": int(self.graph_.number_of_edges()),
                "selected_iteration": search_result.selected_iteration,
                "selected_by": search_result.selected_by,
                "warm_start": self.config.search.warm_start,
                "selected_start": search_result.selected_start,
                "candidate_starts": list(search_result.candidate_starts),
                "start_summaries": list(search_result.start_summaries),
                "dataset_regime": search_result.dataset_regime,
                "prune_profile": search_result.prune_profile,
                "max_pruned_edges": search_result.max_pruned_edges,
                "min_edges_after_prune": search_result.min_edges_after_prune,
                "pre_prune_num_edges": search_result.pre_prune_num_edges,
                "post_prune_num_edges": search_result.post_prune_num_edges,
                "pruned_edges_removed": list(search_result.pruned_edges_removed),
                "operations": list(search_result.operations),
                "score_history": list(search_result.score_history),
                "validation_history": list(search_result.validation_history),
            },
            runtimes={
                "preprocess_seconds": t1 - t0,
                "screener_seconds": t2 - t1,
                "search_seconds": t3 - t2,
                "fit_bn_seconds": t4 - t3,
                "total_seconds": t4 - t0,
            },
            split_metadata={},
            graph_edges=list(self.graph_.edges()),
            candidate_pools=screener_result.candidate_pools,
            screener_diagnostics={
                "candidate_pool_reduction": candidate_pool_reduction(
                    screener_result.candidate_pools,
                    self.graph_.nodes(),
                ),
                "bundle_count": int(sum(len(bundles) for bundles in screener_result.interaction_bundles.values())),
                "score_mask_chunk_size": int(self.config.screener.score_mask_chunk_size),
                "screening_strategy": self.config.screener.strategy,
            },
            screener=screener_result,
            search=search_result,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(frame)
        indices = probabilities.argmax(axis=1)
        if self.target_categories_:
            return np.asarray([self.target_categories_[idx] for idx in indices])
        return indices

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        transformed = self.preprocessor_.transform_for_prediction(frame)
        features = transformed.drop(columns=[self.config.target_column], errors="ignore")
        return self.model_.predict_proba(features)

    def evaluate(self, frame: pd.DataFrame, truth_graph: nx.DiGraph | None = None) -> ExperimentResult:
        if self.model_ is None or self.graph_ is None or self.result_ is None:
            raise RuntimeError("Model must be fit before evaluation.")
        t0 = perf_counter()
        transformed = self.preprocessor_.transform_for_prediction(frame)
        probabilities = self.model_.predict_proba(transformed.drop(columns=[self.config.target_column]))
        t1 = perf_counter()
        predictive = predictive_metrics(
            transformed[self.config.target_column].to_numpy(dtype=int),
            probabilities,
        )
        structural = self._structure_metrics(
            truth_graph=truth_graph,
            predicted=self.graph_,
            candidate_pools=self.result_.candidate_pools,
        )
        return ExperimentResult(
            metadata=self.result_.metadata,
            predictive=predictive,
            structural=structural,
            search_stats=self.result_.search_stats,
            runtimes={**self.result_.runtimes, "evaluation_seconds": t1 - t0},
            split_metadata=self.result_.split_metadata,
            graph_edges=self.result_.graph_edges,
            candidate_pools=self.result_.candidate_pools,
            screener_diagnostics=self.result_.screener_diagnostics,
            screener=self.result_.screener,
            search=self.result_.search,
        )

    def _structure_metrics(
        self,
        truth_graph: nx.DiGraph | None,
        predicted: nx.DiGraph,
        candidate_pools: Dict[str, list[str]],
    ) -> Dict[str, float] | None:
        metrics = {
            "candidate_pool_reduction": candidate_pool_reduction(candidate_pools, predicted.nodes()),
            "predicted_edge_count": float(predicted.number_of_edges()),
        }
        if truth_graph is None:
            return metrics
        metrics["shd"] = float(structural_hamming_distance(predicted, truth_graph))
        metrics["candidate_parent_recall"] = candidate_pool_parent_recall(candidate_pools, truth_graph)
        metrics.update(edge_precision_recall_f1(predicted, truth_graph))
        return metrics

    @staticmethod
    def _full_screener_result(columns: pd.Index) -> ScreenerResult:
        column_names = list(columns)
        zero_matrix = pd.DataFrame(
            np.zeros((len(column_names), len(column_names)), dtype=float),
            index=column_names,
            columns=column_names,
        )
        pools = {
            target: [source for source in column_names if source != target]
            for target in column_names
        }
        return ScreenerResult(
            edge_scores=zero_matrix.copy(),
            edge_stability=zero_matrix.copy(),
            candidate_scores=zero_matrix.copy(),
            candidate_pools=pools,
            interaction_bundles={},
        )

    def _search_candidate_support(self, screener_result: ScreenerResult) -> pd.DataFrame:
        support = pd.DataFrame(
            np.zeros_like(screener_result.candidate_scores.to_numpy(dtype=float)),
            index=screener_result.candidate_scores.index,
            columns=screener_result.candidate_scores.columns,
            dtype=float,
        )
        for target, pool in screener_result.candidate_pools.items():
            positive_scores = [
                float(screener_result.candidate_scores.loc[target, source])
                for source in pool
                if float(screener_result.candidate_scores.loc[target, source]) > 0.0
            ]
            max_positive = max(positive_scores) if positive_scores else 0.0
            for source in pool:
                raw_score = float(screener_result.candidate_scores.loc[target, source])
                normalized_score = raw_score / max_positive if max_positive > 0.0 else 0.0
                stability = float(screener_result.edge_stability.loc[target, source])
                support.loc[target, source] = 0.7 * normalized_score + 0.3 * stability
        return support

    def _log_progress(self, message: str) -> None:
        if not (self.config.screener.progress or self.config.search.progress):
            return
        print(f"[pipeline] {message}", file=sys.stderr, flush=True)
