"""Metrics for structure, calibration, and candidate restriction."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def structural_hamming_distance(
    predicted: nx.DiGraph,
    truth: nx.DiGraph,
    nodes: Iterable[str] | None = None,
) -> int:
    nodes = list(nodes or sorted(set(predicted.nodes()) | set(truth.nodes())))
    shd = 0
    for left, right in product(nodes, nodes):
        if left == right:
            continue
        pred = predicted.has_edge(left, right)
        gt = truth.has_edge(left, right)
        if pred != gt:
            shd += 1
    return shd // 2


def edge_precision_recall_f1(predicted: nx.DiGraph, truth: nx.DiGraph) -> dict[str, float]:
    pred_edges = set(predicted.edges())
    true_edges = set(truth.edges())
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def candidate_pool_reduction(candidate_pools: dict[str, list[str]], nodes: Iterable[str]) -> float:
    nodes = list(nodes)
    full = len(nodes) * (len(nodes) - 1)
    kept = sum(len(parents) for parents in candidate_pools.values())
    if full == 0:
        return 0.0
    return 1.0 - (kept / full)


def candidate_pool_parent_recall(candidate_pools: dict[str, list[str]], truth: nx.DiGraph) -> float:
    true_edges = list(truth.edges())
    if not true_edges:
        return 0.0
    covered = sum(1 for parent, child in true_edges if parent in candidate_pools.get(child, []))
    return covered / len(true_edges)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    if probabilities.ndim == 1:
        confidences = probabilities
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
    y_true = np.asarray(y_true)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high if high < 1.0 else confidences <= high)
        if not np.any(mask):
            continue
        accuracy = correct[mask].mean()
        confidence = confidences[mask].mean()
        ece += np.abs(accuracy - confidence) * mask.mean()
    return float(ece)


def predictive_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    labels = np.arange(probabilities.shape[1], dtype=int)
    metrics = {
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "ece": expected_calibration_error(y_true, probabilities),
    }
    if probabilities.shape[1] == 2:
        metrics["roc_auc"] = _safe_roc_auc_binary(y_true, probabilities[:, 1])
        metrics["brier"] = float(brier_score_loss(y_true, probabilities[:, 1]))
    else:
        metrics["roc_auc"] = _safe_roc_auc_multiclass(y_true, probabilities, labels=labels)
        one_hot = np.eye(probabilities.shape[1])[y_true]
        metrics["brier"] = float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))
    return metrics


def _safe_roc_auc_binary(y_true: np.ndarray, positive_probabilities: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        try:
            score = roc_auc_score(y_true, positive_probabilities)
        except ValueError:
            return 0.5
    if np.isnan(score):
        return 0.5
    return float(score)


def _safe_roc_auc_multiclass(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        try:
            score = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        except ValueError:
            return 0.5
    if np.isnan(score):
        return 0.5
    return float(score)
