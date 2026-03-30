"""Synthetic data generators for structure-learning smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticDataset:
    frame: pd.DataFrame
    graph: nx.DiGraph
    target_column: str


def make_mixed_classification_dataset(
    n_samples: int = 500,
    n_features: int = 6,
    random_state: int = 7,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_state)
    feature_names = [f"x{i}" for i in range(n_features)]
    target = "label"
    graph = nx.DiGraph()
    graph.add_nodes_from(feature_names + [target])

    order = feature_names + [target]
    for i, child in enumerate(order):
        for parent in order[:i]:
            if rng.random() < 0.25:
                graph.add_edge(parent, child)

    data: dict[str, np.ndarray] = {}
    for index, name in enumerate(feature_names):
        parents = list(graph.predecessors(name))
        base = rng.normal(0.0, 1.0, size=n_samples)
        if parents:
            signal = sum(0.7 * np.tanh(np.asarray(data[parent], dtype=float)) for parent in parents)
            base += signal
        if index % 2 == 0:
            probs = 1.0 / (1.0 + np.exp(-base))
            values = (rng.random(n_samples) < probs).astype(int)
            data[name] = values
        else:
            data[name] = base + rng.normal(0.0, 0.5, size=n_samples)

    parent_signal = np.zeros(n_samples, dtype=float)
    for parent in graph.predecessors(target):
        parent_signal += 0.8 * np.asarray(data[parent], dtype=float)
    logits = parent_signal + rng.normal(0.0, 0.6, size=n_samples)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    data[target] = np.where(rng.random(n_samples) < probabilities, "yes", "no")

    frame = pd.DataFrame(data)
    return SyntheticDataset(frame=frame, graph=graph, target_column=target)
