from __future__ import annotations

import networkx as nx
import numpy as np

from neural_bn.metrics import (
    candidate_pool_reduction,
    edge_precision_recall_f1,
    expected_calibration_error,
    predictive_metrics,
    structural_hamming_distance,
)


def test_structure_metrics_behave_sensibly():
    truth = nx.DiGraph([("a", "b"), ("b", "c")])
    pred = nx.DiGraph([("a", "b"), ("c", "b")])

    shd = structural_hamming_distance(pred, truth, nodes=["a", "b", "c"])
    pr = edge_precision_recall_f1(pred, truth)

    assert shd >= 1
    assert 0.0 <= pr["precision"] <= 1.0
    assert 0.0 <= pr["recall"] <= 1.0
    assert 0.0 <= candidate_pool_reduction({"a": [], "b": ["a"], "c": ["a", "b"]}, ["a", "b", "c"]) <= 1.0


def test_calibration_error_zero_for_perfect_binary_confidence():
    y_true = np.array([0, 1, 0, 1])
    probs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert expected_calibration_error(y_true, probs) == 0.0


def test_predictive_metrics_handles_multiclass_validation_missing_a_class():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=float,
    )

    metrics = predictive_metrics(y_true, probs)

    assert metrics["log_loss"] >= 0.0
    assert metrics["brier"] >= 0.0
    assert metrics["roc_auc"] == 0.5


def test_predictive_metrics_handles_binary_validation_with_one_class():
    y_true = np.array([1, 1, 1], dtype=int)
    probs = np.array(
        [
            [0.20, 0.80],
            [0.10, 0.90],
            [0.30, 0.70],
        ],
        dtype=float,
    )

    metrics = predictive_metrics(y_true, probs)

    assert metrics["log_loss"] >= 0.0
    assert metrics["brier"] >= 0.0
    assert metrics["roc_auc"] == 0.5
