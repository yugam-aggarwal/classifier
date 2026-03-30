from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from neural_bn.config import ScreenerConfig, SearchConfig
from neural_bn.data import TabularPreprocessor
from neural_bn.models import DiscreteBayesianNetwork
from neural_bn.screener import InteractionBundle, NeuralDependencyScreener, _MultiTargetNet
from neural_bn.search import ConstrainedBNStructureLearner, GraphScorer, SearchResult


def _reference_masked_loss(model, x, target, columns, target_idx, extra_masks):
    masked = x.clone()
    target_info = columns[target_idx]
    masked[:, target_info.input_indices] = 0.0
    for source_idx in extra_masks:
        masked[:, columns[source_idx].input_indices] = 0.0
    output = model.forward_head(masked, target_idx)
    if target_info.kind == "continuous":
        loss = torch.nn.functional.mse_loss(output.squeeze(-1), target.float())
    else:
        loss = torch.nn.functional.cross_entropy(output, target.long())
    return float(loss.item())


def _prepared_small_frame():
    frame = pd.DataFrame(
        {
            "cat_a": ["x", "y", "x", "z", "y", "z"],
            "num_b": [0.1, 0.2, 0.9, 1.0, 0.3, 0.8],
            "num_c": [1.2, 0.4, 1.0, 0.2, 1.4, 0.1],
            "target": ["no", "no", "yes", "yes", "no", "yes"],
        }
    )
    preprocessor = TabularPreprocessor(target_column="target")
    return preprocessor.fit_transform(frame)


def test_adaptive_candidate_pool_retains_bundle_sources():
    config = ScreenerConfig(
        candidate_pool_size=1,
        candidate_pool_soft_cap=3,
        candidate_pool_score_ratio=0.8,
        force_include_bundle_sources=True,
    )
    screener = NeuralDependencyScreener(config)
    columns = ["target", "a", "b", "c"]
    edge_scores = pd.DataFrame(
        [
            [0.0, 1.0, 0.79, 0.2],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=columns,
        columns=columns,
    )
    edge_stability = pd.DataFrame(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=columns,
        columns=columns,
    )
    bundles = {
        "target": [
            InteractionBundle(
                target="target",
                sources=("b", "c"),
                score=0.6,
                stability=1.0,
            )
        ]
    }

    candidate_scores, candidate_pools = screener._candidate_pools(columns, edge_scores, edge_stability, bundles)

    assert candidate_pools["target"] == ["b", "a", "c"]
    assert candidate_scores.loc["target", "b"] > candidate_scores.loc["target", "a"]


def test_batched_training_loss_matches_reference_loop():
    prepared = _prepared_small_frame()
    config = ScreenerConfig(hidden_dim=8, num_layers=1, dropout=0.0, score_mask_chunk_size=2)
    screener = NeuralDependencyScreener(config)
    x = torch.tensor(prepared.encoded_matrix, dtype=torch.float32)
    targets = {
        column.name: torch.tensor(prepared.target_arrays[column.name])
        for column in prepared.columns
    }
    keep_masks = screener._build_keep_masks(prepared.columns, x.shape[1], device=x.device, dtype=x.dtype)
    torch.manual_seed(0)
    model = _MultiTargetNet(x.shape[1], prepared.columns, config)
    model.train()

    batch_x = x[:4]
    batch_targets = {
        column.name: targets[column.name][:4]
        for column in prepared.columns
    }
    batched_loss = screener._batched_training_loss(
        model=model,
        batch_x=batch_x,
        targets=batch_targets,
        columns=prepared.columns,
        keep_masks=keep_masks,
    )
    reference_loss = torch.tensor(0.0)
    for target_idx, column in enumerate(prepared.columns):
        masked = batch_x.clone()
        masked[:, column.input_indices] = 0.0
        output = model.forward_head(masked, target_idx)
        target = batch_targets[column.name]
        if column.kind == "continuous":
            reference_loss = reference_loss + torch.nn.functional.mse_loss(output.squeeze(-1), target.float())
        else:
            reference_loss = reference_loss + torch.nn.functional.cross_entropy(output, target.long())
    reference_loss = reference_loss / len(prepared.columns)

    assert torch.isclose(batched_loss, reference_loss, atol=1e-6)


def test_batched_scoring_matches_reference_mask_loop():
    prepared = _prepared_small_frame()
    config = ScreenerConfig(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        pairwise_top_m=2,
        interaction_top_pairs=2,
        score_mask_chunk_size=2,
    )
    screener = NeuralDependencyScreener(config)
    x = torch.tensor(prepared.encoded_matrix, dtype=torch.float32)
    targets = {
        column.name: torch.tensor(prepared.target_arrays[column.name])
        for column in prepared.columns
    }
    keep_masks = screener._build_keep_masks(prepared.columns, x.shape[1], device=x.device, dtype=x.dtype)
    torch.manual_seed(0)
    model = _MultiTargetNet(x.shape[1], prepared.columns, config)
    model.eval()

    edge_matrix, bundle_scores = screener._score_model(
        model=model,
        x=x,
        targets=targets,
        columns=prepared.columns,
        keep_masks=keep_masks,
    )

    reference_edge_matrix = np.zeros_like(edge_matrix)
    reference_bundle_scores = {}
    name_to_idx = {column.name: idx for idx, column in enumerate(prepared.columns)}
    with torch.no_grad():
        for target_idx, target_info in enumerate(prepared.columns):
            baseline = _reference_masked_loss(
                model,
                x,
                targets[target_info.name],
                prepared.columns,
                target_idx,
                (),
            )
            single_scores = {}
            for source_idx, source_info in enumerate(prepared.columns):
                if source_idx == target_idx:
                    continue
                loss = _reference_masked_loss(
                    model,
                    x,
                    targets[target_info.name],
                    prepared.columns,
                    target_idx,
                    (source_idx,),
                )
                influence = max(0.0, (loss - baseline) / max(abs(baseline), 1e-6))
                reference_edge_matrix[target_idx, source_idx] = influence
                single_scores[source_info.name] = influence

            top_sources = sorted(single_scores, key=single_scores.get, reverse=True)[: config.pairwise_top_m]
            for left, right in combinations(top_sources, 2):
                loss = _reference_masked_loss(
                    model,
                    x,
                    targets[target_info.name],
                    prepared.columns,
                    target_idx,
                    (name_to_idx[left], name_to_idx[right]),
                )
                relative_increase = max(0.0, (loss - baseline) / max(abs(baseline), 1e-6))
                synergy = max(0.0, relative_increase - single_scores[left] - single_scores[right])
                reference_bundle_scores[(target_info.name, tuple(sorted((left, right))))] = synergy

    assert np.allclose(edge_matrix, reference_edge_matrix, atol=1e-6)
    assert set(bundle_scores) == set(reference_bundle_scores)
    for key in reference_bundle_scores:
        assert bundle_scores[key] == pytest.approx(reference_bundle_scores[key], abs=1e-6)


def test_target_markov_blanket_score_matches_full_bn_conditional():
    frame = pd.DataFrame(
        {
            "x": [0, 0, 1, 1, 0, 1, 0, 1],
            "a": [0, 1, 0, 1, 1, 0, 1, 0],
            "target": [0, 0, 1, 1, 1, 0, 1, 0],
        }
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(frame.columns)
    graph.add_edge("target", "a")
    graph.add_edge("x", "a")

    config = SearchConfig(generative_weight=0.0, discriminative_weight=1.0)
    scorer = GraphScorer(frame, "target", config)
    mb_score = scorer.discriminative_score(graph)

    bn = DiscreteBayesianNetwork("target", laplace=config.laplace).fit(frame, graph)
    probabilities = bn.predict_proba(frame.drop(columns=["target"]))
    y_true = frame["target"].to_numpy(dtype=int)
    full_score = float(np.sum(np.log(probabilities[np.arange(len(y_true)), y_true] + 1e-12)))

    assert np.isclose(mb_score, full_score)


def test_search_limits_ranked_additions_and_keeps_deletions():
    config = SearchConfig(max_additions_per_child=1, warm_start="empty")
    searcher = ConstrainedBNStructureLearner(config)
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("a", "b")

    candidate_pools = {
        "a": ["c", "b"],
        "b": [],
        "c": [],
    }
    candidate_scores = pd.DataFrame(
        [
            [0.0, 0.4, 0.9],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    proposals = list(searcher._iter_proposals(graph, candidate_pools, candidate_scores))
    op_names = [name for _, name in proposals]

    assert "add c->a" in op_names
    assert "delete a->b" in op_names
    assert "reverse a->b" not in op_names


def test_screener_sparse_warm_start_is_acyclic_and_not_target_fanout():
    config = SearchConfig(warm_start="screener_sparse")
    searcher = ConstrainedBNStructureLearner(config)
    columns = ["target", "a", "b", "c"]
    candidate_pools = {
        "target": ["a", "b"],
        "a": ["target", "b"],
        "b": ["target", "a"],
        "c": ["target"],
    }
    candidate_scores = pd.DataFrame(
        [
            [0.0, 0.9, 0.4, 0.0],
            [0.8, 0.0, 0.7, 0.0],
            [0.6, 0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
        ],
        index=columns,
        columns=columns,
    )

    graph = searcher._initialize_graph(columns, "target", candidate_pools, candidate_scores)

    assert nx.is_directed_acyclic_graph(graph)
    assert graph.out_degree("target") < 3
    assert max(graph.in_degree(node) for node in graph.nodes()) <= 1


def test_graph_score_is_stable_under_row_duplication_with_normalization():
    frame = pd.DataFrame(
        {
            "a": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )
    duplicated = pd.concat([frame, frame], ignore_index=True)
    graph = nx.DiGraph()
    graph.add_nodes_from(frame.columns)
    graph.add_edge("a", "target")
    candidate_support = pd.DataFrame(
        [[0.0, 0.0], [0.8, 0.0]],
        index=["a", "target"],
        columns=["a", "target"],
    )
    normalized = SearchConfig(normalize_objective_by_samples=True)
    raw = SearchConfig(normalize_objective_by_samples=False)

    first_normalized = GraphScorer(frame, "target", normalized, candidate_support=candidate_support).graph_score(graph)
    second_normalized = GraphScorer(duplicated, "target", normalized, candidate_support=candidate_support).graph_score(graph)
    first_raw = GraphScorer(frame, "target", raw, candidate_support=candidate_support).graph_score(graph)
    second_raw = GraphScorer(duplicated, "target", raw, candidate_support=candidate_support).graph_score(graph)

    assert abs(first_normalized[0] - second_normalized[0]) < abs(first_raw[0] - second_raw[0])


def test_pruning_removes_weak_edge_after_checkpoint_selection(monkeypatch):
    config = SearchConfig(
        warm_start="empty",
        max_iters=2,
        max_additions_per_child=1,
        validation_checkpoint_interval=1,
        validation_prune_passes=1,
        validation_prune_passes_real=1,
        validation_prune_min_edges_real=0,
        validation_prune_max_fraction_real=1.0,
    )
    searcher = ConstrainedBNStructureLearner(config)
    frame = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 0, 1],
            "b": [0, 1, 0, 1, 0, 1],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    candidate_pools = {"a": ["target"], "b": ["a"], "target": []}
    candidate_scores = pd.DataFrame(0.0, index=frame.columns, columns=frame.columns)
    candidate_scores.loc["a", "target"] = 1.0
    candidate_scores.loc["b", "a"] = 0.1

    def fake_graph_score(self, graph):
        objective = 0.0
        if graph.has_edge("target", "a"):
            objective += 10.0
        if graph.has_edge("a", "b"):
            objective += 1.0
        return objective, objective, 0.0, 0.0, 0.0

    checkpoints = {
        0: {"log_loss": 0.9, "roc_auc": 0.8},
        1: {"log_loss": 0.2, "roc_auc": 0.95},
        2: {"log_loss": 0.2, "roc_auc": 0.95},
    }

    def fake_checkpoint(*, graph, iteration, reason, scorer, **_kwargs):
        metrics = checkpoints[iteration]
        total, gen, disc, prior, hub = scorer.graph_score(graph)
        return {
            "iteration": iteration,
            "reason": reason,
            "log_loss": metrics["log_loss"],
            "roc_auc": metrics["roc_auc"],
            "num_edges": int(graph.number_of_edges()),
            "objective_total": total,
            "generative_score": gen,
            "discriminative_score": disc,
            "prior_score": prior,
            "hub_penalty": hub,
        }

    def fake_predictive_metrics(y_true, probabilities):
        if probabilities.shape[1] == 2 and probabilities[0, 0] > 0.8:
            return {"log_loss": 0.2, "roc_auc": 0.95, "brier": 0.1, "ece": 0.0}
        return {"log_loss": 0.1995, "roc_auc": 0.95, "brier": 0.1, "ece": 0.0}

    def fake_predict_proba(self, frame):
        confidence = 0.9 if self.graph_.number_of_edges() > 1 else 0.7
        probs = np.tile(np.array([confidence, 1.0 - confidence]), (len(frame), 1))
        return probs

    monkeypatch.setattr(GraphScorer, "graph_score", fake_graph_score)
    monkeypatch.setattr(searcher, "_evaluate_validation_checkpoint", fake_checkpoint)
    monkeypatch.setattr("neural_bn.search.predictive_metrics", fake_predictive_metrics)
    monkeypatch.setattr(DiscreteBayesianNetwork, "predict_proba", fake_predict_proba)
    result = searcher.fit(
        frame=frame,
        target_column="target",
        candidate_pools=candidate_pools,
        candidate_scores=candidate_scores,
        validation_frame=frame,
        dataset_regime="real",
    )

    assert result.post_prune_num_edges < result.pre_prune_num_edges
    assert result.pruned_edges_removed
    assert result.selected_by == "validation_log_loss_pruned"


def test_real_regime_pruning_stops_at_configured_floor(monkeypatch):
    columns = ["target"] + [f"x{i}" for i in range(12)]
    frame = pd.DataFrame(
        {
            column: ([0, 1] * 8)[:16]
            for column in columns
        }
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for idx in range(len(columns) - 1):
        graph.add_edge(columns[idx], columns[idx + 1])

    config = SearchConfig(
        validation_prune_passes_real=1,
        validation_prune_min_edges_real=0,
        validation_prune_max_fraction_real=0.35,
    )
    searcher = ConstrainedBNStructureLearner(config)
    candidate_support = pd.DataFrame(0.0, index=columns, columns=columns)
    scorer = GraphScorer(frame, "target", config, candidate_support=candidate_support)
    starting_validation = {
        "iteration": 12,
        "reason": "selected",
        "log_loss": 0.5,
        "roc_auc": 0.8,
        "num_edges": graph.number_of_edges(),
        "objective_total": float(-graph.number_of_edges()),
        "generative_score": float(-graph.number_of_edges()),
        "discriminative_score": 0.0,
        "prior_score": 0.0,
        "hub_penalty": 0.0,
    }

    def fake_graph_score(self, graph):
        edge_count = float(graph.number_of_edges())
        return -edge_count, -edge_count, 0.0, 0.0, 0.0

    def fake_predictive_metrics(y_true, probabilities):
        return {"log_loss": 0.5, "roc_auc": 0.8, "brier": 0.1, "ece": 0.0}

    def fake_predict_proba(self, frame):
        return np.tile(np.array([0.5, 0.5]), (len(frame), 1))

    monkeypatch.setattr(GraphScorer, "graph_score", fake_graph_score)
    monkeypatch.setattr("neural_bn.search.predictive_metrics", fake_predictive_metrics)
    monkeypatch.setattr(DiscreteBayesianNetwork, "predict_proba", fake_predict_proba)

    pruned_graph, _, pruned_edges_removed, _ = searcher._prune_selected_graph(
        train_frame=frame,
        validation_frame=frame,
        target_column="target",
        graph=graph,
        scorer=scorer,
        validation_history=[starting_validation],
        starting_validation=starting_validation,
        dataset_regime="real",
    )

    assert pruned_graph.number_of_edges() == 8
    assert len(pruned_edges_removed) == 4


def test_high_support_target_edge_requires_strict_log_loss_improvement(monkeypatch):
    frame = pd.DataFrame(
        {
            "a": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(frame.columns)
    graph.add_edge("a", "target")

    config = SearchConfig(
        validation_prune_passes_real=1,
        validation_prune_min_edges_real=0,
        validation_prune_max_fraction_real=1.0,
    )
    searcher = ConstrainedBNStructureLearner(config)
    candidate_support = pd.DataFrame(
        [[0.0, 0.0], [0.9, 0.0]],
        index=["a", "target"],
        columns=["a", "target"],
    )
    scorer = GraphScorer(frame, "target", config, candidate_support=candidate_support)
    scorer.edge_local_generative_contribution = lambda graph, parent, child: -1.0
    starting_validation = {
        "iteration": 1,
        "reason": "selected",
        "log_loss": 0.3,
        "roc_auc": 0.8,
        "num_edges": graph.number_of_edges(),
        "objective_total": float(-graph.number_of_edges()),
        "generative_score": float(-graph.number_of_edges()),
        "discriminative_score": 0.0,
        "prior_score": 0.0,
        "hub_penalty": 0.0,
    }

    def fake_graph_score(self, graph):
        edge_count = float(graph.number_of_edges())
        return -edge_count, -edge_count, 0.0, 0.0, 0.0

    def fake_predictive_metrics(y_true, probabilities):
        return {"log_loss": 0.3, "roc_auc": 0.8, "brier": 0.1, "ece": 0.0}

    def fake_predict_proba(self, frame):
        return np.tile(np.array([0.6, 0.4]), (len(frame), 1))

    monkeypatch.setattr(GraphScorer, "graph_score", fake_graph_score)
    monkeypatch.setattr("neural_bn.search.predictive_metrics", fake_predictive_metrics)
    monkeypatch.setattr(DiscreteBayesianNetwork, "predict_proba", fake_predict_proba)

    pruned_graph, _, pruned_edges_removed, _ = searcher._prune_selected_graph(
        train_frame=frame,
        validation_frame=frame,
        target_column="target",
        graph=graph,
        scorer=scorer,
        validation_history=[starting_validation],
        starting_validation=starting_validation,
        dataset_regime="real",
    )

    assert pruned_graph.has_edge("a", "target")
    assert not pruned_edges_removed


def test_pruning_reranks_candidates_after_each_acceptance(monkeypatch):
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "a": [0, 1, 0, 1, 0, 1],
            "b": [1, 0, 1, 0, 1, 0],
            "c": [0, 0, 1, 1, 0, 1],
        }
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(frame.columns)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")

    config = SearchConfig(
        validation_prune_passes_real=1,
        validation_prune_min_edges_real=0,
        validation_prune_max_fraction_real=1.0,
    )
    searcher = ConstrainedBNStructureLearner(config)
    scorer = GraphScorer(
        frame,
        "target",
        config,
        candidate_support=pd.DataFrame(0.0, index=frame.columns, columns=frame.columns),
    )
    starting_validation = {
        "iteration": 2,
        "reason": "selected",
        "log_loss": 0.4,
        "roc_auc": 0.8,
        "num_edges": graph.number_of_edges(),
        "objective_total": float(-graph.number_of_edges()),
        "generative_score": float(-graph.number_of_edges()),
        "discriminative_score": 0.0,
        "prior_score": 0.0,
        "hub_penalty": 0.0,
    }
    rank_calls: list[tuple[tuple[str, str], ...]] = []

    def fake_graph_score(self, graph):
        edge_count = float(graph.number_of_edges())
        return -edge_count, -edge_count, 0.0, 0.0, 0.0

    def fake_predictive_metrics(y_true, probabilities):
        return {"log_loss": 0.4, "roc_auc": 0.8, "brier": 0.1, "ece": 0.0}

    def fake_predict_proba(self, frame):
        return np.tile(np.array([0.55, 0.45]), (len(frame), 1))

    def fake_rank_prune_edges(graph, scorer):
        rank_calls.append(tuple(sorted(graph.edges())))
        return sorted(graph.edges())

    monkeypatch.setattr(GraphScorer, "graph_score", fake_graph_score)
    monkeypatch.setattr("neural_bn.search.predictive_metrics", fake_predictive_metrics)
    monkeypatch.setattr(DiscreteBayesianNetwork, "predict_proba", fake_predict_proba)
    monkeypatch.setattr(searcher, "_rank_prune_edges", fake_rank_prune_edges)

    pruned_graph, _, pruned_edges_removed, _ = searcher._prune_selected_graph(
        train_frame=frame,
        validation_frame=frame,
        target_column="target",
        graph=graph,
        scorer=scorer,
        validation_history=[starting_validation],
        starting_validation=starting_validation,
        dataset_regime="real",
    )

    assert len(rank_calls) >= 2
    assert rank_calls[0] != rank_calls[1]
    assert len(pruned_edges_removed) == 2
    assert pruned_graph.number_of_edges() == 0
    assert nx.is_directed_acyclic_graph(pruned_graph)


def test_validation_checkpoint_can_select_earlier_graph(monkeypatch):
    config = SearchConfig(
        warm_start="empty",
        max_iters=2,
        max_additions_per_child=1,
        validation_checkpoint_interval=1,
    )
    searcher = ConstrainedBNStructureLearner(config)
    frame = pd.DataFrame(
        {
            "a": [0, 1, 0, 1],
            "b": [1, 0, 1, 0],
            "target": [0, 1, 0, 1],
        }
    )
    candidate_pools = {"a": ["target"], "b": ["a"], "target": []}
    candidate_scores = pd.DataFrame(
        0.0,
        index=frame.columns,
        columns=frame.columns,
    )
    candidate_scores.loc["a", "target"] = 1.0
    candidate_scores.loc["b", "a"] = 1.0

    def fake_graph_score(self, graph):
        edge_count = graph.number_of_edges()
        return float(edge_count), float(edge_count), 0.0, 0.0, 0.0

    def fake_checkpoint(*, graph, iteration, reason, scorer, **_kwargs):
        losses = {0: 0.9, 1: 0.1, 2: 0.5}
        total, gen, disc, prior, hub = scorer.graph_score(graph)
        return {
            "iteration": iteration,
            "reason": reason,
            "log_loss": losses[iteration],
            "roc_auc": 0.9 - 0.1 * iteration,
            "num_edges": int(graph.number_of_edges()),
            "objective_total": total,
            "generative_score": gen,
            "discriminative_score": disc,
            "prior_score": prior,
            "hub_penalty": hub,
        }

    monkeypatch.setattr(GraphScorer, "graph_score", fake_graph_score)
    monkeypatch.setattr(searcher, "_evaluate_validation_checkpoint", fake_checkpoint)

    result = searcher.fit(
        frame=frame,
        target_column="target",
        candidate_pools=candidate_pools,
        candidate_scores=candidate_scores,
        validation_frame=frame,
    )

    assert result.selected_iteration == 1
    assert result.graph.number_of_edges() == 1
    assert result.last_graph.number_of_edges() == 2


def _mock_search_result(
    *,
    start: str,
    objective_total: float,
    num_edges: int,
    log_loss: float | None,
    roc_auc: float | None,
    selected_by: str = "validation_log_loss",
) -> SearchResult:
    graph = nx.DiGraph()
    graph.add_nodes_from(["target", "a", "b"])
    for index in range(num_edges):
        graph.add_edge("target", f"n{index}")
    summary = {
        "start": start,
        "selected_iteration": 3,
        "selected_by": selected_by,
        "num_edges": num_edges,
        "objective_total": objective_total,
        "runtime_seconds": 1.0,
        "log_loss": log_loss,
        "roc_auc": roc_auc,
    }
    return SearchResult(
        graph=graph,
        last_graph=graph.copy(),
        total_score=objective_total,
        generative_score=objective_total,
        discriminative_score=0.0,
        prior_score=0.0,
        hub_penalty=0.0,
        score_history=[objective_total],
        operations=[],
        runtime_seconds=1.0,
        selected_iteration=3,
        selected_by=selected_by,
        validation_history=[],
        pruned_edges_removed=[],
        pre_prune_num_edges=num_edges,
        post_prune_num_edges=num_edges,
        dataset_regime="unknown",
        max_pruned_edges=0,
        min_edges_after_prune=0,
        prune_profile="real",
        selected_start=start,
        candidate_starts=[start],
        start_summaries=[summary],
    )


def test_multi_start_selects_lower_validation_log_loss(monkeypatch):
    config = SearchConfig(
        warm_start="multi_start",
        multi_start_candidates=("naive_bayes", "screener_sparse"),
    )
    searcher = ConstrainedBNStructureLearner(config)
    frame = pd.DataFrame({"target": [0, 1], "a": [0, 1]})

    def fake_fit_single_start(*, warm_start, **_kwargs):
        if warm_start == "naive_bayes":
            return _mock_search_result(
                start=warm_start,
                objective_total=0.6,
                num_edges=5,
                log_loss=0.20,
                roc_auc=0.90,
            )
        return _mock_search_result(
            start=warm_start,
            objective_total=0.4,
            num_edges=6,
            log_loss=0.10,
            roc_auc=0.80,
        )

    monkeypatch.setattr(searcher, "_fit_single_start", fake_fit_single_start)

    result = searcher.fit(
        frame=frame,
        target_column="target",
        candidate_pools={"target": ["a"], "a": []},
        candidate_scores=None,
        validation_frame=frame,
    )

    assert result.selected_start == "screener_sparse"
    assert result.candidate_starts == ["naive_bayes", "screener_sparse"]
    assert len(result.start_summaries) == 2


def test_multi_start_tie_breaks_by_start_order_after_metrics_and_objective(monkeypatch):
    config = SearchConfig(
        warm_start="multi_start",
        multi_start_candidates=("naive_bayes", "screener_sparse"),
    )
    searcher = ConstrainedBNStructureLearner(config)
    frame = pd.DataFrame({"target": [0, 1], "a": [0, 1]})

    def fake_fit_single_start(*, warm_start, **_kwargs):
        return _mock_search_result(
            start=warm_start,
            objective_total=0.5,
            num_edges=4,
            log_loss=0.10,
            roc_auc=0.90,
        )

    monkeypatch.setattr(searcher, "_fit_single_start", fake_fit_single_start)

    result = searcher.fit(
        frame=frame,
        target_column="target",
        candidate_pools={"target": ["a"], "a": []},
        candidate_scores=None,
        validation_frame=frame,
    )

    assert result.selected_start == "naive_bayes"


def test_multi_start_without_validation_selects_better_objective(monkeypatch):
    config = SearchConfig(
        warm_start="multi_start",
        multi_start_candidates=("naive_bayes", "screener_sparse"),
    )
    searcher = ConstrainedBNStructureLearner(config)
    frame = pd.DataFrame({"target": [0, 1], "a": [0, 1]})

    def fake_fit_single_start(*, warm_start, **_kwargs):
        if warm_start == "naive_bayes":
            return _mock_search_result(
                start=warm_start,
                objective_total=0.5,
                num_edges=5,
                log_loss=None,
                roc_auc=None,
                selected_by="last_iteration",
            )
        return _mock_search_result(
            start=warm_start,
            objective_total=0.7,
            num_edges=6,
            log_loss=None,
            roc_auc=None,
            selected_by="last_iteration",
        )

    monkeypatch.setattr(searcher, "_fit_single_start", fake_fit_single_start)

    result = searcher.fit(
        frame=frame,
        target_column="target",
        candidate_pools={"target": ["a"], "a": []},
        candidate_scores=None,
        validation_frame=None,
    )

    assert result.selected_start == "screener_sparse"
