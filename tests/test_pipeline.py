from __future__ import annotations

import networkx as nx

from neural_bn.config import PipelineConfig
from neural_bn.pipeline import NeuralScreenedBNClassifier
from neural_bn.synthetic import make_mixed_classification_dataset


def test_pipeline_smoke_runs_end_to_end():
    dataset = make_mixed_classification_dataset(n_samples=140, n_features=5, random_state=13)
    config = PipelineConfig(target_column=dataset.target_column)
    config.screener.epochs = 3
    config.screener.bootstrap_rounds = 2
    config.screener.candidate_pool_size = 3
    config.search.max_iters = 8

    model = NeuralScreenedBNClassifier(config).fit(dataset.frame, truth_graph=dataset.graph)

    assert model.graph_ is not None
    assert nx.is_directed_acyclic_graph(model.graph_)
    assert set(model.result_.screener.candidate_pools) == set(dataset.frame.columns)
    assert "log_loss" in model.result_.predictive
    assert "candidate_pool_reduction" in model.result_.structural
    assert model.result_.screener_diagnostics["score_mask_chunk_size"] == config.screener.score_mask_chunk_size


def test_predict_proba_works_without_target_column():
    dataset = make_mixed_classification_dataset(n_samples=120, n_features=4, random_state=21)
    config = PipelineConfig(target_column=dataset.target_column)
    config.screener.epochs = 2
    config.screener.bootstrap_rounds = 1
    config.search.max_iters = 5

    model = NeuralScreenedBNClassifier(config).fit(dataset.frame)
    features_only = dataset.frame.drop(columns=[dataset.target_column])
    probabilities = model.predict_proba(features_only)

    assert probabilities.shape[0] == len(features_only)
    assert probabilities.shape[1] == 2


def test_pipeline_reproducibility_with_same_seed():
    dataset = make_mixed_classification_dataset(n_samples=120, n_features=4, random_state=21)
    config = PipelineConfig(target_column=dataset.target_column)
    config.random_state = 99
    config.screener.random_state = 99
    config.screener.epochs = 2
    config.screener.bootstrap_rounds = 1
    config.search.max_iters = 5

    first = NeuralScreenedBNClassifier(config).fit(dataset.frame)
    second = NeuralScreenedBNClassifier(config).fit(dataset.frame)

    assert first.result_.graph_edges == second.result_.graph_edges
    assert first.result_.candidate_pools == second.result_.candidate_pools
