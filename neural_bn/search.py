"""Candidate-constrained BN structure search."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .config import SearchConfig
from .metrics import predictive_metrics
from .models import DiscreteBayesianNetwork


@dataclass(slots=True)
class SearchResult:
    graph: nx.DiGraph
    last_graph: nx.DiGraph
    total_score: float
    generative_score: float
    discriminative_score: float
    prior_score: float
    hub_penalty: float
    score_history: List[float]
    operations: List[str]
    runtime_seconds: float
    selected_iteration: int
    selected_by: str
    validation_history: List[Dict[str, Any]]
    pruned_edges_removed: List[str]
    pre_prune_num_edges: int
    post_prune_num_edges: int
    dataset_regime: str
    max_pruned_edges: int
    min_edges_after_prune: int
    prune_profile: str
    selected_start: str
    candidate_starts: List[str]
    start_summaries: List[Dict[str, Any]]


class GraphScorer:
    def __init__(
        self,
        frame: pd.DataFrame,
        target_column: str,
        config: SearchConfig,
        candidate_support: pd.DataFrame | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).astype(int)
        self.target_column = target_column
        self.config = config
        self.n_samples = len(frame)
        columns = list(self.frame.columns)
        self.cardinalities = {column: int(self.frame[column].max()) + 1 for column in self.frame.columns}
        self.local_cache: Dict[tuple[str, tuple[str, ...]], float] = {}
        self.count_cache: Dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self.probability_cache: Dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self.discriminative_cache: Dict[tuple[Any, ...], float] = {}
        self.column_arrays = {
            column: self.frame[column].to_numpy(dtype=int)
            for column in self.frame.columns
        }
        if candidate_support is None:
            self.candidate_support = pd.DataFrame(
                np.zeros((len(columns), len(columns)), dtype=float),
                index=columns,
                columns=columns,
            )
        else:
            self.candidate_support = candidate_support.reindex(
                index=columns,
                columns=columns,
                fill_value=0.0,
            ).astype(float)

    def graph_score(self, graph: nx.DiGraph) -> tuple[float, float, float, float, float]:
        generative = 0.0
        for node in self.frame.columns:
            parents = tuple(sorted(graph.predecessors(node)))
            generative += self.local_score(node, parents)
        discriminative = self.discriminative_score(graph) if self.config.discriminative_weight else 0.0
        generative_component = self._normalized_component(generative)
        discriminative_component = self._normalized_component(discriminative)
        prior_score = self.graph_prior(graph)
        hub_penalty = self.graph_hub_penalty(graph)
        total = (
            self.config.generative_weight * generative_component
            + self.config.discriminative_weight * discriminative_component
            + self.config.edge_prior_weight * prior_score
            - self.config.hub_penalty_weight * hub_penalty
        )
        return total, generative_component, discriminative_component, prior_score, hub_penalty

    def local_score(self, node: str, parents: tuple[str, ...]) -> float:
        key = (node, parents)
        if key in self.local_cache:
            return self.local_cache[key]
        counts = self._counts(node, list(parents))
        if self.config.generative_score.lower() == "bdeu":
            score = self._bdeu_score(counts)
        else:
            score = self._bic_score(counts)
        self.local_cache[key] = score
        return score

    def discriminative_score(self, graph: nx.DiGraph) -> float:
        signature = self._target_markov_blanket_signature(graph)
        if signature in self.discriminative_cache:
            return self.discriminative_cache[signature]
        score = self._target_markov_blanket_log_likelihood(graph)
        self.discriminative_cache[signature] = score
        return score

    def family_probabilities(self, node: str, parents: tuple[str, ...]) -> np.ndarray:
        key = (node, parents)
        if key in self.probability_cache:
            return self.probability_cache[key]
        counts = self._counts(node, list(parents))
        probabilities = (counts + self.config.laplace) / (
            counts.sum(axis=-1, keepdims=True) + self.config.laplace * counts.shape[-1]
        )
        self.probability_cache[key] = probabilities
        return probabilities

    def edge_support(self, parent: str, child: str) -> float:
        if child not in self.candidate_support.index or parent not in self.candidate_support.columns:
            return 0.0
        return float(self.candidate_support.loc[child, parent])

    def graph_prior(self, graph: nx.DiGraph) -> float:
        if graph.number_of_edges() == 0:
            return 0.0
        prior_values = [
            2.0 * self.edge_support(parent, child) - 1.0
            for parent, child in graph.edges()
        ]
        return float(np.mean(prior_values)) if prior_values else 0.0

    def graph_hub_penalty(self, graph: nx.DiGraph) -> float:
        node_count = max(graph.number_of_nodes(), 1)
        if node_count == 0:
            return 0.0
        excess_out_degree = [
            max(graph.out_degree(node) - 2, 0)
            for node in graph.nodes()
        ]
        return float(np.mean(excess_out_degree) / max(node_count, 1))

    def edge_local_generative_contribution(self, graph: nx.DiGraph, parent: str, child: str) -> float:
        parents = tuple(sorted(graph.predecessors(child)))
        if parent not in parents:
            return 0.0
        reduced = tuple(node for node in parents if node != parent)
        return float(self.local_score(child, parents) - self.local_score(child, reduced))

    def _normalized_component(self, value: float) -> float:
        if not self.config.normalize_objective_by_samples:
            return float(value)
        return float(value / max(self.n_samples, 1))

    def _counts(self, node: str, parents: List[str]) -> np.ndarray:
        key = (node, tuple(parents))
        if key in self.count_cache:
            return self.count_cache[key]
        node_cardinality = self.cardinalities[node]
        node_values = self.column_arrays[node]
        if not parents:
            counts = np.bincount(node_values, minlength=node_cardinality).astype(float)
            self.count_cache[key] = counts
            return counts
        parent_cardinalities = [self.cardinalities[parent] for parent in parents]
        parent_values = tuple(self.column_arrays[parent] for parent in parents)
        parent_index = np.ravel_multi_index(parent_values, dims=tuple(parent_cardinalities))
        combined_index = parent_index * node_cardinality + node_values
        total_states = int(np.prod(parent_cardinalities, dtype=int)) * node_cardinality
        counts = np.bincount(combined_index, minlength=total_states).astype(float)
        counts = counts.reshape(tuple(parent_cardinalities) + (node_cardinality,))
        self.count_cache[key] = counts
        return counts

    def _bic_score(self, counts: np.ndarray) -> float:
        if counts.ndim == 1:
            row_sums = counts.sum()
            ll = np.sum(counts[counts > 0] * np.log(counts[counts > 0] / row_sums))
            num_params = counts.shape[0] - 1
            return float(ll - 0.5 * num_params * np.log(self.n_samples))
        flat = counts.reshape(-1, counts.shape[-1])
        row_sums = flat.sum(axis=1, keepdims=True)
        probs = np.divide(flat, row_sums, out=np.zeros_like(flat), where=row_sums > 0)
        ll = np.sum(flat[flat > 0] * np.log(probs[flat > 0]))
        q = flat.shape[0]
        r = flat.shape[1]
        num_params = q * (r - 1)
        return float(ll - 0.5 * num_params * np.log(self.n_samples))

    def _bdeu_score(self, counts: np.ndarray) -> float:
        flat = counts.reshape(-1, counts.shape[-1]) if counts.ndim > 1 else counts.reshape(1, -1)
        q = flat.shape[0]
        r = flat.shape[1]
        alpha_ij = self.config.equivalent_sample_size / max(q, 1)
        alpha_ijk = alpha_ij / r
        row_sums = flat.sum(axis=1)
        score = np.sum(gammaln(alpha_ij) - gammaln(alpha_ij + row_sums))
        score += np.sum(gammaln(alpha_ijk + flat) - gammaln(alpha_ijk))
        return float(score)

    def _target_markov_blanket_signature(self, graph: nx.DiGraph) -> tuple[Any, ...]:
        target_parents = tuple(sorted(graph.predecessors(self.target_column)))
        child_families = tuple(
            sorted(
                (child, tuple(sorted(graph.predecessors(child))))
                for child in graph.successors(self.target_column)
            )
        )
        return target_parents, child_families

    def _target_markov_blanket_log_likelihood(self, graph: nx.DiGraph) -> float:
        y_true = self.column_arrays[self.target_column]
        y_card = self.cardinalities[self.target_column]
        classes = np.arange(y_card, dtype=int)
        n_samples = len(y_true)

        target_parents = tuple(sorted(graph.predecessors(self.target_column)))
        target_probabilities = self.family_probabilities(self.target_column, target_parents)
        log_scores = np.log(
            self._target_probability_matrix(target_probabilities, target_parents, classes, n_samples) + 1e-12
        )

        for child in sorted(graph.successors(self.target_column)):
            child_parents = tuple(sorted(graph.predecessors(child)))
            child_probabilities = self.family_probabilities(child, child_parents)
            child_values = self.column_arrays[child]
            flat_probabilities = child_probabilities.reshape(-1, child_probabilities.shape[-1])
            for class_idx, class_value in enumerate(classes):
                parent_index = self._parent_index(
                    child_parents,
                    target_override=np.full(n_samples, class_value, dtype=int),
                )
                log_scores[:, class_idx] += np.log(flat_probabilities[parent_index, child_values] + 1e-12)

        return float(np.sum(log_scores[np.arange(n_samples), y_true] - logsumexp(log_scores, axis=1)))

    def _target_probability_matrix(
        self,
        probabilities: np.ndarray,
        parents: tuple[str, ...],
        classes: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if not parents:
            return np.broadcast_to(probabilities[classes], (n_samples, len(classes)))
        parent_index = self._parent_index(parents)
        flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        return flat_probabilities[parent_index][:, classes]

    def _parent_index(
        self,
        parents: tuple[str, ...],
        target_override: np.ndarray | None = None,
    ) -> np.ndarray:
        parent_values = tuple(
            target_override if parent == self.target_column and target_override is not None else self.column_arrays[parent]
            for parent in parents
        )
        parent_cardinalities = tuple(self.cardinalities[parent] for parent in parents)
        if not parent_values:
            return np.zeros(self.n_samples, dtype=int)
        return np.ravel_multi_index(parent_values, dims=parent_cardinalities)


class ConstrainedBNStructureLearner:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def fit(
        self,
        frame: pd.DataFrame,
        target_column: str,
        candidate_pools: Dict[str, List[str]],
        candidate_scores: pd.DataFrame | None = None,
        validation_frame: pd.DataFrame | None = None,
        dataset_regime: str = "unknown",
    ) -> SearchResult:
        if self.config.warm_start != "multi_start":
            return self._fit_single_start(
                frame=frame,
                target_column=target_column,
                candidate_pools=candidate_pools,
                candidate_scores=candidate_scores,
                validation_frame=validation_frame,
                dataset_regime=dataset_regime,
                warm_start=self.config.warm_start,
            )

        candidate_starts = self._candidate_starts()
        start_results: List[SearchResult] = []
        validation_available = validation_frame is not None
        multi_start_begin = perf_counter()
        for index, warm_start in enumerate(candidate_starts, start=1):
            self._log_progress(
                f"multi-start {index}/{len(candidate_starts)} begin start={warm_start}"
            )
            result = self._fit_single_start(
                frame=frame,
                target_column=target_column,
                candidate_pools=candidate_pools,
                candidate_scores=candidate_scores,
                validation_frame=validation_frame,
                dataset_regime=dataset_regime,
                warm_start=warm_start,
            )
            start_results.append(result)

        selected_index = self._select_multi_start_result(start_results, validation_available)
        total_runtime = perf_counter() - multi_start_begin
        winner = start_results[selected_index]
        winner.runtime_seconds = total_runtime
        winner.selected_start = candidate_starts[selected_index]
        winner.candidate_starts = list(candidate_starts)
        winner.start_summaries = [
            dict(result.start_summaries[0])
            for result in start_results
        ]
        self._log_progress(
            f"multi-start selected start={winner.selected_start} "
            f"runtime={total_runtime:.1f}s edges={winner.graph.number_of_edges()}"
        )
        return winner

    def _fit_single_start(
        self,
        *,
        frame: pd.DataFrame,
        target_column: str,
        candidate_pools: Dict[str, List[str]],
        candidate_scores: pd.DataFrame | None,
        validation_frame: pd.DataFrame | None,
        dataset_regime: str,
        warm_start: str,
    ) -> SearchResult:
        graph = self._initialize_graph(
            frame.columns,
            target_column,
            candidate_pools,
            candidate_scores,
            warm_start=warm_start,
        )
        scorer = GraphScorer(frame, target_column, self.config, candidate_support=candidate_scores)
        validation_data = validation_frame.reset_index(drop=True).astype(int) if validation_frame is not None else None
        start_time = perf_counter()
        total, generative, discriminative, prior_score, hub_penalty = scorer.graph_score(graph)
        history = [total]
        operations: List[str] = []
        validation_history: List[Dict[str, Any]] = []
        selected_graph = graph.copy()
        selected_iteration = 0
        selected_by = "last_iteration"
        best_validation: Dict[str, Any] | None = None
        no_improvement_checkpoints = 0
        small_delta_streak = 0
        pruned_edges_removed: List[str] = []
        prune_profile = self._prune_profile(dataset_regime)
        min_edges_after_prune = 0
        max_pruned_edges = 0
        pre_prune_num_edges = int(graph.number_of_edges())
        post_prune_num_edges = int(graph.number_of_edges())
        self._log_progress(
            "starting search "
            f"nodes={len(frame.columns)} edges={graph.number_of_edges()} "
            f"max_iters={self.config.max_iters}"
        )
        self._log_progress(
            "initial score "
            f"total={total:.3f} gen={generative:.3f} disc={discriminative:.3f} "
            f"prior={prior_score:.3f} hub={hub_penalty:.3f}"
        )
        if validation_data is not None:
            checkpoint = self._evaluate_validation_checkpoint(
                train_frame=frame,
                validation_frame=validation_data,
                target_column=target_column,
                graph=graph,
                iteration=0,
                reason="initial",
                scorer=scorer,
            )
            validation_history.append(checkpoint)
            best_validation = checkpoint
            selected_graph = graph.copy()
            selected_by = "validation_log_loss"

        for iteration_idx in range(self.config.max_iters):
            iteration_number = iteration_idx + 1
            best_move: tuple[float, nx.DiGraph, str, float, float, float, float] | None = None
            proposal_count = 0
            for proposal_count, (proposal, op_name) in enumerate(
                self._iter_proposals(graph, candidate_pools, candidate_scores),
                start=1,
            ):
                proposal_total, proposal_gen, proposal_disc, proposal_prior, proposal_hub = scorer.graph_score(proposal)
                delta = proposal_total - total
                if delta > self.config.improve_tol and (
                    best_move is None or delta > best_move[0]
                ):
                    best_move = (
                        delta,
                        proposal,
                        op_name,
                        proposal_gen,
                        proposal_disc,
                        proposal_prior,
                        proposal_hub,
                    )
                self._maybe_log_proposal(iteration_number, proposal_count, best_move)
            if best_move is None:
                self._log_progress(
                    f"iteration {iteration_number}/{self.config.max_iters} "
                    f"no improving move after proposals={proposal_count}"
                )
                break
            delta, graph, op_name, generative, discriminative, prior_score, hub_penalty = best_move
            total = (
                self.config.generative_weight * generative
                + self.config.discriminative_weight * discriminative
                + self.config.edge_prior_weight * prior_score
                - self.config.hub_penalty_weight * hub_penalty
            )
            history.append(total)
            operations.append(op_name)
            delta_per_sample = delta if self.config.normalize_objective_by_samples else delta / max(scorer.n_samples, 1)
            if delta_per_sample < self.config.min_delta_per_sample:
                small_delta_streak += 1
            else:
                small_delta_streak = 0
            self._log_progress(
                f"iteration {iteration_number}/{self.config.max_iters} accepted "
                f"move={op_name} proposals={proposal_count} score={total:.3f} "
                f"edges={graph.number_of_edges()}"
            )
            if validation_data is not None and len(operations) % self.config.validation_checkpoint_interval == 0:
                checkpoint = self._evaluate_validation_checkpoint(
                    train_frame=frame,
                    validation_frame=validation_data,
                    target_column=target_column,
                    graph=graph,
                    iteration=len(operations),
                    reason="interval",
                    scorer=scorer,
                )
                validation_history.append(checkpoint)
                if self._is_better_checkpoint(checkpoint, best_validation):
                    best_validation = checkpoint
                    selected_graph = graph.copy()
                    selected_iteration = len(operations)
                    selected_by = "validation_log_loss"
                    no_improvement_checkpoints = 0
                else:
                    no_improvement_checkpoints += 1
                if (
                    small_delta_streak >= self.config.validation_patience
                    and no_improvement_checkpoints >= self.config.validation_patience
                ):
                    self._log_progress(
                        f"early stop after iteration {iteration_number}: "
                        f"small_delta_streak={small_delta_streak} "
                        f"validation_patience={no_improvement_checkpoints}"
                    )
                    break

        last_graph = graph.copy()
        if validation_data is not None:
            final_iteration = len(operations)
            if not validation_history or validation_history[-1]["iteration"] != final_iteration:
                checkpoint = self._evaluate_validation_checkpoint(
                    train_frame=frame,
                    validation_frame=validation_data,
                    target_column=target_column,
                    graph=graph,
                    iteration=final_iteration,
                    reason="final",
                    scorer=scorer,
                )
                validation_history.append(checkpoint)
                if self._is_better_checkpoint(checkpoint, best_validation):
                    best_validation = checkpoint
                    selected_graph = graph.copy()
                    selected_iteration = final_iteration
                    selected_by = "validation_log_loss"
        else:
            selected_graph = graph.copy()
            selected_iteration = len(operations)
            selected_by = "last_iteration"

        pre_prune_num_edges = int(selected_graph.number_of_edges())
        (
            prune_profile,
            _prune_passes,
            _prune_log_loss_tol,
            _prune_roc_auc_tol,
            min_edges_after_prune,
            max_pruned_edges,
        ) = self._prune_settings(
            dataset_regime=dataset_regime,
            pre_prune_num_edges=pre_prune_num_edges,
        )
        if validation_data is not None and _prune_passes > 0 and selected_graph.number_of_edges() > 0:
            (
                selected_graph,
                best_validation,
                pruned_edges_removed,
                validation_history,
            ) = self._prune_selected_graph(
                train_frame=frame,
                validation_frame=validation_data,
                target_column=target_column,
                graph=selected_graph,
                scorer=scorer,
                validation_history=validation_history,
                starting_validation=best_validation,
                dataset_regime=dataset_regime,
            )
            if pruned_edges_removed:
                selected_by = "validation_log_loss_pruned"
        post_prune_num_edges = int(selected_graph.number_of_edges())

        selected_total, selected_generative, selected_discriminative, selected_prior, selected_hub = scorer.graph_score(selected_graph)

        runtime = perf_counter() - start_time
        start_summary = {
            "start": warm_start,
            "selected_iteration": int(selected_iteration),
            "selected_by": selected_by,
            "num_edges": int(selected_graph.number_of_edges()),
            "objective_total": float(selected_total),
            "runtime_seconds": float(runtime),
            "log_loss": float(best_validation["log_loss"]) if best_validation is not None else None,
            "roc_auc": float(best_validation["roc_auc"]) if best_validation is not None else None,
        }
        self._log_progress(
            f"finished search start={warm_start} iterations={len(history) - 1} "
            f"selected_edges={selected_graph.number_of_edges()} runtime={runtime:.1f}s"
        )
        return SearchResult(
            graph=selected_graph,
            last_graph=last_graph,
            total_score=selected_total,
            generative_score=selected_generative,
            discriminative_score=selected_discriminative,
            prior_score=selected_prior,
            hub_penalty=selected_hub,
            score_history=history,
            operations=operations,
            runtime_seconds=runtime,
            selected_iteration=selected_iteration,
            selected_by=selected_by,
            validation_history=validation_history,
            pruned_edges_removed=pruned_edges_removed,
            pre_prune_num_edges=pre_prune_num_edges,
            post_prune_num_edges=post_prune_num_edges,
            dataset_regime=dataset_regime,
            max_pruned_edges=max_pruned_edges,
            min_edges_after_prune=min_edges_after_prune,
            prune_profile=prune_profile,
            selected_start=warm_start,
            candidate_starts=[warm_start],
            start_summaries=[start_summary],
        )

    def _initialize_graph(
        self,
        columns: Iterable[str],
        target_column: str,
        candidate_pools: Dict[str, List[str]],
        candidate_scores: pd.DataFrame | None,
        warm_start: str | None = None,
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(columns)
        start_mode = warm_start or self.config.warm_start
        if start_mode == "multi_start":
            start_mode = self._candidate_starts()[0]
        if start_mode == "empty":
            return graph
        if start_mode == "naive_bayes":
            for child in columns:
                if child == target_column:
                    continue
                if target_column in candidate_pools.get(child, []) and graph.in_degree(child) < self.config.max_parents:
                    graph.add_edge(target_column, child)
            return graph
        if start_mode != "screener_sparse":
            return graph

        seed_edges: List[tuple[float, str, str]] = []
        for child in columns:
            pool = candidate_pools.get(child, [])
            if not pool:
                continue
            ranked = sorted(
                (
                    (self._candidate_score(child, parent, candidate_scores), parent)
                    for parent in pool
                    if parent != child
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if not ranked or ranked[0][0] <= 0:
                continue
            support, parent = ranked[0]
            seed_edges.append((support, parent, child))

        for _, parent, child in sorted(seed_edges, key=lambda item: item[0], reverse=True):
            if graph.in_degree(child) > 0:
                continue
            proposal = graph.copy()
            proposal.add_edge(parent, child)
            if nx.is_directed_acyclic_graph(proposal):
                graph = proposal
        return graph

    def _candidate_starts(self) -> List[str]:
        if self.config.warm_start != "multi_start":
            return [self.config.warm_start]
        candidate_starts = list(dict.fromkeys(self.config.multi_start_candidates))
        if not candidate_starts:
            raise ValueError("multi_start_candidates must contain at least one warm start.")
        return candidate_starts

    def _select_multi_start_result(
        self,
        start_results: List[SearchResult],
        validation_available: bool,
    ) -> int:
        selected_index = 0
        for index in range(1, len(start_results)):
            candidate_summary = start_results[index].start_summaries[0]
            best_summary = start_results[selected_index].start_summaries[0]
            if self._is_better_start_summary(candidate_summary, best_summary, validation_available):
                selected_index = index
        return selected_index

    @staticmethod
    def _is_better_start_summary(
        candidate: Dict[str, Any],
        best: Dict[str, Any],
        validation_available: bool,
    ) -> bool:
        if validation_available:
            candidate_log_loss = float(candidate["log_loss"])
            best_log_loss = float(best["log_loss"])
            if candidate_log_loss < best_log_loss - 1e-12:
                return True
            if (
                abs(candidate_log_loss - best_log_loss) <= 1e-12
                and float(candidate["roc_auc"]) > float(best["roc_auc"]) + 1e-12
            ):
                return True
            if (
                abs(candidate_log_loss - best_log_loss) <= 1e-12
                and abs(float(candidate["roc_auc"]) - float(best["roc_auc"])) <= 1e-12
                and float(candidate["objective_total"]) > float(best["objective_total"]) + 1e-12
            ):
                return True
            if (
                abs(candidate_log_loss - best_log_loss) <= 1e-12
                and abs(float(candidate["roc_auc"]) - float(best["roc_auc"])) <= 1e-12
                and abs(float(candidate["objective_total"]) - float(best["objective_total"])) <= 1e-12
                and int(candidate["num_edges"]) < int(best["num_edges"])
            ):
                return True
            return False
        if float(candidate["objective_total"]) > float(best["objective_total"]) + 1e-12:
            return True
        if (
            abs(float(candidate["objective_total"]) - float(best["objective_total"])) <= 1e-12
            and int(candidate["num_edges"]) < int(best["num_edges"])
        ):
            return True
        return False

    def _iter_proposals(
        self,
        graph: nx.DiGraph,
        candidate_pools: Dict[str, List[str]],
        candidate_scores: pd.DataFrame | None = None,
    ) -> Iterable[tuple[nx.DiGraph, str]]:
        add_candidates: List[tuple[float, str, str]] = []
        for child in graph.nodes():
            current_parents = set(graph.predecessors(child))
            if len(current_parents) >= self.config.max_parents:
                continue
            for parent in self._ranked_additions(child, graph, candidate_pools, candidate_scores):
                add_candidates.append((self._candidate_score(child, parent, candidate_scores), parent, child))

        for _, parent, child in sorted(add_candidates, key=lambda item: item[0], reverse=True):
            proposal = graph.copy()
            proposal.add_edge(parent, child)
            if nx.is_directed_acyclic_graph(proposal):
                yield proposal, f"add {parent}->{child}"

        for parent, child in list(graph.edges()):
            proposal = graph.copy()
            proposal.remove_edge(parent, child)
            yield proposal, f"delete {parent}->{child}"

        reversal_candidates: List[tuple[float, str, str]] = []
        for parent, child in list(graph.edges()):
            if child not in self._ranked_additions(parent, graph, candidate_pools, candidate_scores):
                continue
            if graph.in_degree(parent) + 1 > self.config.max_parents:
                continue
            reversal_candidates.append((self._candidate_score(parent, child, candidate_scores), parent, child))

        for _, parent, child in sorted(reversal_candidates, key=lambda item: item[0], reverse=True):
            proposal = graph.copy()
            proposal.remove_edge(parent, child)
            proposal.add_edge(child, parent)
            if nx.is_directed_acyclic_graph(proposal):
                yield proposal, f"reverse {parent}->{child}"

    def _ranked_additions(
        self,
        child: str,
        graph: nx.DiGraph,
        candidate_pools: Dict[str, List[str]],
        candidate_scores: pd.DataFrame | None,
    ) -> List[str]:
        current_parents = set(graph.predecessors(child))
        ranked = [
            parent
            for parent in candidate_pools.get(child, [])
            if parent != child and parent not in current_parents
        ]
        ranked.sort(key=lambda parent: self._candidate_score(child, parent, candidate_scores), reverse=True)
        return ranked[: self.config.max_additions_per_child]

    @staticmethod
    def _candidate_score(child: str, parent: str, candidate_scores: pd.DataFrame | None) -> float:
        if candidate_scores is None:
            return 0.0
        if child not in candidate_scores.index or parent not in candidate_scores.columns:
            return 0.0
        return float(candidate_scores.loc[child, parent])

    def _evaluate_validation_checkpoint(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        target_column: str,
        graph: nx.DiGraph,
        iteration: int,
        reason: str,
        scorer: GraphScorer,
    ) -> Dict[str, Any]:
        model = DiscreteBayesianNetwork(target_column=target_column, laplace=self.config.laplace).fit(
            train_frame,
            graph,
        )
        probabilities = model.predict_proba(validation_frame.drop(columns=[target_column]))
        metrics = predictive_metrics(validation_frame[target_column].to_numpy(dtype=int), probabilities)
        total_score, generative_score, discriminative_score, prior_score, hub_penalty = scorer.graph_score(graph)
        checkpoint = {
            "iteration": int(iteration),
            "reason": reason,
            "log_loss": float(metrics["log_loss"]),
            "roc_auc": float(metrics["roc_auc"]),
            "num_edges": int(graph.number_of_edges()),
            "objective_total": float(total_score),
            "generative_score": float(generative_score),
            "discriminative_score": float(discriminative_score),
            "prior_score": float(prior_score),
            "hub_penalty": float(hub_penalty),
        }
        self._log_progress(
            f"validation iteration={checkpoint['iteration']} "
            f"log_loss={checkpoint['log_loss']:.4f} roc_auc={checkpoint['roc_auc']:.4f} "
            f"edges={checkpoint['num_edges']} objective={checkpoint['objective_total']:.4f}"
        )
        return checkpoint

    @staticmethod
    def _is_better_checkpoint(
        candidate: Dict[str, Any],
        best: Dict[str, Any] | None,
    ) -> bool:
        if best is None:
            return True
        if candidate["log_loss"] < best["log_loss"] - 1e-12:
            return True
        if abs(candidate["log_loss"] - best["log_loss"]) <= 1e-12 and candidate["roc_auc"] > best["roc_auc"] + 1e-12:
            return True
        if (
            abs(candidate["log_loss"] - best["log_loss"]) <= 1e-12
            and abs(candidate["roc_auc"] - best["roc_auc"]) <= 1e-12
            and candidate["objective_total"] > best["objective_total"] + 1e-12
        ):
            return True
        if (
            abs(candidate["log_loss"] - best["log_loss"]) <= 1e-12
            and abs(candidate["roc_auc"] - best["roc_auc"]) <= 1e-12
            and abs(candidate["objective_total"] - best["objective_total"]) <= 1e-12
            and candidate["num_edges"] < best["num_edges"]
        ):
            return True
        return False

    def _prune_selected_graph(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        target_column: str,
        graph: nx.DiGraph,
        scorer: GraphScorer,
        validation_history: List[Dict[str, Any]],
        starting_validation: Dict[str, Any] | None,
        dataset_regime: str,
    ) -> tuple[nx.DiGraph, Dict[str, Any] | None, List[str], List[Dict[str, Any]]]:
        (
            prune_profile,
            prune_passes,
            _prune_log_loss_tol,
            _prune_roc_auc_tol,
            min_edges_after_prune,
            max_pruned_edges,
        ) = self._prune_settings(
            dataset_regime=dataset_regime,
            pre_prune_num_edges=int(graph.number_of_edges()),
        )
        current_graph = graph.copy()
        current_model = DiscreteBayesianNetwork(target_column=target_column, laplace=self.config.laplace).fit(
            train_frame,
            current_graph,
        )
        probabilities = current_model.predict_proba(validation_frame.drop(columns=[target_column]))
        current_metrics = predictive_metrics(validation_frame[target_column].to_numpy(dtype=int), probabilities)
        current_total, current_gen, current_disc, current_prior, current_hub = scorer.graph_score(current_graph)
        current_validation = {
            "iteration": int(validation_history[-1]["iteration"]) if validation_history else 0,
            "reason": "selected",
            "log_loss": float(current_metrics["log_loss"]),
            "roc_auc": float(current_metrics["roc_auc"]),
            "num_edges": int(current_graph.number_of_edges()),
            "objective_total": float(current_total),
            "generative_score": float(current_gen),
            "discriminative_score": float(current_disc),
            "prior_score": float(current_prior),
            "hub_penalty": float(current_hub),
        }
        pruned_edges_removed: List[str] = []

        for pass_idx in range(prune_passes):
            if current_graph.number_of_edges() <= min_edges_after_prune:
                break
            if len(pruned_edges_removed) >= max_pruned_edges:
                break
            accepted_this_pass = 0
            while True:
                if current_graph.number_of_edges() <= min_edges_after_prune:
                    break
                if len(pruned_edges_removed) >= max_pruned_edges:
                    break
                accepted_edge = False
                for parent, child in self._rank_prune_edges(current_graph, scorer):
                    if not current_graph.has_edge(parent, child):
                        continue
                    if current_graph.number_of_edges() <= min_edges_after_prune:
                        break
                    if len(pruned_edges_removed) >= max_pruned_edges:
                        break
                    proposal_graph = current_graph.copy()
                    proposal_graph.remove_edge(parent, child)
                    proposal_model = current_model.copy().update_local_cpds(train_frame, proposal_graph, [child])
                    proposal_probabilities = proposal_model.predict_proba(validation_frame.drop(columns=[target_column]))
                    proposal_metrics = predictive_metrics(
                        validation_frame[target_column].to_numpy(dtype=int),
                        proposal_probabilities,
                    )
                    proposal_total, proposal_gen, proposal_disc, proposal_prior, proposal_hub = scorer.graph_score(
                        proposal_graph
                    )
                    requires_strict_improvement = self._prune_acceptance_requires_strict_improvement(
                        parent=parent,
                        child=child,
                        target_column=target_column,
                        scorer=scorer,
                    )
                    if not self._accept_pruned_graph(
                        candidate_metrics=proposal_metrics,
                        current_metrics=current_metrics,
                        candidate_total=proposal_total,
                        current_total=current_total,
                        prune_profile=prune_profile,
                        require_strict_log_loss_improvement=requires_strict_improvement,
                    ):
                        continue
                    current_graph = proposal_graph
                    current_model = proposal_model
                    current_metrics = proposal_metrics
                    current_total = proposal_total
                    current_gen = proposal_gen
                    current_disc = proposal_disc
                    current_prior = proposal_prior
                    current_hub = proposal_hub
                    pruned_edges_removed.append(f"delete {parent}->{child}")
                    accepted_this_pass += 1
                    accepted_edge = True
                    current_validation = {
                        "iteration": current_validation["iteration"],
                        "reason": f"prune_{prune_profile}_pass_{pass_idx + 1}",
                        "log_loss": float(current_metrics["log_loss"]),
                        "roc_auc": float(current_metrics["roc_auc"]),
                        "num_edges": int(current_graph.number_of_edges()),
                        "objective_total": float(current_total),
                        "generative_score": float(current_gen),
                        "discriminative_score": float(current_disc),
                        "prior_score": float(current_prior),
                        "hub_penalty": float(current_hub),
                    }
                    validation_history.append(current_validation)
                    self._log_progress(
                        f"prune profile={prune_profile} pass={pass_idx + 1} "
                        f"accepted delete {parent}->{child} "
                        f"log_loss={current_validation['log_loss']:.4f} "
                        f"roc_auc={current_validation['roc_auc']:.4f} "
                        f"edges={current_validation['num_edges']}"
                    )
                    break
                if not accepted_edge:
                    break
            if accepted_this_pass == 0:
                break
        if not pruned_edges_removed:
            return current_graph, starting_validation, pruned_edges_removed, validation_history
        return current_graph, current_validation if pruned_edges_removed else starting_validation, pruned_edges_removed, validation_history

    def _rank_prune_edges(
        self,
        graph: nx.DiGraph,
        scorer: GraphScorer,
    ) -> List[tuple[str, str]]:
        ranked_edges: List[tuple[float, float, int, str, str]] = []
        for parent, child in graph.edges():
            support = scorer.edge_support(parent, child)
            local_contribution = scorer.edge_local_generative_contribution(graph, parent, child)
            if support >= self.config.validation_prune_support_protect and local_contribution > 0.0:
                continue
            ranked_edges.append(
                (
                    support,
                    local_contribution,
                    -graph.out_degree(parent),
                    parent,
                    child,
                )
            )
        ranked_edges.sort()
        return [(parent, child) for _, _, _, parent, child in ranked_edges]

    def _accept_pruned_graph(
        self,
        *,
        candidate_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        candidate_total: float,
        current_total: float,
        prune_profile: str,
        require_strict_log_loss_improvement: bool,
    ) -> bool:
        candidate_log_loss = float(candidate_metrics["log_loss"])
        current_log_loss = float(current_metrics["log_loss"])
        candidate_roc_auc = float(candidate_metrics["roc_auc"])
        current_roc_auc = float(current_metrics["roc_auc"])
        if candidate_log_loss < current_log_loss - 1e-12:
            return True
        if require_strict_log_loss_improvement:
            return False
        if prune_profile == "known_graph":
            log_loss_tol = self.config.validation_prune_log_loss_tol
            roc_auc_tol = self.config.validation_prune_roc_auc_tol
        else:
            log_loss_tol = self.config.validation_prune_log_loss_tol_real
            roc_auc_tol = self.config.validation_prune_roc_auc_tol_real
        return (
            candidate_log_loss <= current_log_loss + log_loss_tol
            and candidate_roc_auc >= current_roc_auc - roc_auc_tol
            and candidate_total > current_total + 1e-4
        )

    @staticmethod
    def _prune_profile(dataset_regime: str) -> str:
        if dataset_regime == "known_graph":
            return "known_graph"
        return "real"

    def _prune_settings(
        self,
        *,
        dataset_regime: str,
        pre_prune_num_edges: int,
    ) -> tuple[str, int, float, float, int, int]:
        prune_profile = self._prune_profile(dataset_regime)
        if prune_profile == "known_graph":
            prune_passes = max(self.config.validation_prune_passes, 0)
            log_loss_tol = self.config.validation_prune_log_loss_tol
            roc_auc_tol = self.config.validation_prune_roc_auc_tol
            max_fraction = float(np.clip(self.config.validation_prune_max_fraction_known_graph, 0.0, 1.0))
            min_edges_after_prune = max(
                self.config.validation_prune_min_edges_known_graph,
                int(np.ceil((1.0 - max_fraction) * pre_prune_num_edges)),
            )
        else:
            prune_passes = max(self.config.validation_prune_passes_real, 0)
            log_loss_tol = self.config.validation_prune_log_loss_tol_real
            roc_auc_tol = self.config.validation_prune_roc_auc_tol_real
            max_fraction = float(np.clip(self.config.validation_prune_max_fraction_real, 0.0, 1.0))
            min_edges_after_prune = max(
                self.config.validation_prune_min_edges_real,
                int(np.ceil((1.0 - max_fraction) * pre_prune_num_edges)),
            )
        max_pruned_edges = int(np.ceil(max_fraction * pre_prune_num_edges))
        return (
            prune_profile,
            prune_passes,
            log_loss_tol,
            roc_auc_tol,
            int(min_edges_after_prune),
            max_pruned_edges,
        )

    def _prune_acceptance_requires_strict_improvement(
        self,
        *,
        parent: str,
        child: str,
        target_column: str,
        scorer: GraphScorer,
    ) -> bool:
        is_target_adjacent = parent == target_column or child == target_column
        if not is_target_adjacent:
            return False
        return scorer.edge_support(parent, child) >= self.config.validation_prune_target_support_protect

    def _maybe_log_proposal(
        self,
        iteration_number: int,
        proposal_count: int,
        best_move: tuple[float, nx.DiGraph, str, float, float, float, float] | None,
    ) -> None:
        if not self.config.progress:
            return
        interval = max(self.config.proposal_progress_interval, 1)
        if proposal_count % interval != 0:
            return
        best_delta = best_move[0] if best_move is not None else 0.0
        best_name = best_move[2] if best_move is not None else "none"
        self._log_progress(
            f"iteration {iteration_number}/{self.config.max_iters} "
            f"proposals={proposal_count} best_delta={best_delta:.4f} best_move={best_name}"
        )

    def _log_progress(self, message: str) -> None:
        if not self.config.progress:
            return
        print(f"[search] {message}", file=sys.stderr, flush=True)
