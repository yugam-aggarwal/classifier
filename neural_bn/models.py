"""Discrete Bayesian network models and built-in BN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(slots=True)
class LocalDiscreteCPD:
    node: str
    parents: List[str]
    parent_cardinalities: List[int]
    probabilities: np.ndarray


class DiscreteBayesianNetwork:
    """A small explicit discrete BN implementation for classification experiments."""

    def __init__(self, target_column: str, laplace: float = 1.0) -> None:
        self.target_column = target_column
        self.laplace = laplace
        self.graph_: nx.DiGraph | None = None
        self.cardinalities_: Dict[str, int] = {}
        self.cpds_: Dict[str, LocalDiscreteCPD] = {}
        self.columns_: List[str] = []

    def fit(self, frame: pd.DataFrame, graph: nx.DiGraph) -> "DiscreteBayesianNetwork":
        if self.target_column not in frame.columns:
            raise KeyError(f"Target column '{self.target_column}' missing from frame.")
        self.graph_ = nx.DiGraph(graph)
        self.graph_.add_nodes_from(frame.columns)
        self.columns_ = list(frame.columns)
        self.cardinalities_ = {
            column: int(frame[column].max()) + 1 for column in frame.columns
        }
        self.cpds_ = {}
        for node in self.columns_:
            parents = sorted(self.graph_.predecessors(node))
            self.cpds_[node] = self._fit_local_cpd(frame, node, parents)
        return self

    def copy(self) -> "DiscreteBayesianNetwork":
        if self.graph_ is None:
            raise RuntimeError("Model must be fit before copying.")
        cloned = DiscreteBayesianNetwork(self.target_column, laplace=self.laplace)
        cloned.graph_ = nx.DiGraph(self.graph_)
        cloned.cardinalities_ = dict(self.cardinalities_)
        cloned.columns_ = list(self.columns_)
        cloned.cpds_ = {
            node: LocalDiscreteCPD(
                node=cpd.node,
                parents=list(cpd.parents),
                parent_cardinalities=list(cpd.parent_cardinalities),
                probabilities=np.array(cpd.probabilities, copy=True),
            )
            for node, cpd in self.cpds_.items()
        }
        return cloned

    def update_local_cpds(
        self,
        frame: pd.DataFrame,
        graph: nx.DiGraph,
        nodes: Iterable[str],
    ) -> "DiscreteBayesianNetwork":
        if self.graph_ is None:
            raise RuntimeError("Model must be fit before updating CPDs.")
        self.graph_ = nx.DiGraph(graph)
        self.graph_.add_nodes_from(self.columns_)
        for node in nodes:
            parents = sorted(self.graph_.predecessors(node))
            self.cpds_[node] = self._fit_local_cpd(frame, node, parents)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame).argmax(axis=1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        log_probs = self.predict_log_proba(frame)
        shifted = log_probs - log_probs.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict_log_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.graph_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        rows = frame.copy()
        if self.target_column in rows.columns:
            rows = rows.drop(columns=[self.target_column])
        classes = np.arange(self.cardinalities_[self.target_column], dtype=int)
        n_rows = len(rows)
        row_values = {
            column: rows[column].to_numpy(dtype=int)
            for column in rows.columns
        }
        log_probs = np.zeros((len(rows), len(classes)), dtype=float)
        for node in self.columns_:
            cpd = self.cpds_[node]
            if node == self.target_column:
                target_probs = self._target_probabilities(cpd, row_values, classes, n_rows)
                log_probs += np.log(target_probs + 1e-12)
                continue

            node_values = row_values[node]
            if self.target_column not in cpd.parents:
                node_probs = self._lookup_probabilities(cpd, row_values, node_values)
                log_probs += np.log(node_probs + 1e-12)[:, None]
                continue

            for class_idx, class_value in enumerate(classes):
                target_values = np.full(n_rows, class_value, dtype=int)
                node_probs = self._lookup_probabilities(
                    cpd,
                    row_values,
                    node_values,
                    target_values=target_values,
                )
                log_probs[:, class_idx] += np.log(node_probs + 1e-12)
        return log_probs

    def _node_probability(self, node: str, assignment: Dict[str, int]) -> float:
        cpd = self.cpds_[node]
        node_value = int(assignment[node])
        if not cpd.parents:
            return float(cpd.probabilities[node_value])
        index = tuple(int(assignment[parent]) for parent in cpd.parents)
        return float(cpd.probabilities[index + (node_value,)])

    def _counts(self, frame: pd.DataFrame, node: str, parents: List[str]) -> np.ndarray:
        node_cardinality = self.cardinalities_[node]
        node_values = frame[node].to_numpy(dtype=int)
        if not parents:
            return np.bincount(node_values, minlength=node_cardinality).astype(float)

        parent_cardinalities = [self.cardinalities_[parent] for parent in parents]
        parent_values = tuple(frame[parent].to_numpy(dtype=int) for parent in parents)
        parent_index = np.ravel_multi_index(parent_values, dims=tuple(parent_cardinalities))
        combined_index = parent_index * node_cardinality + node_values
        total_states = int(np.prod(parent_cardinalities, dtype=int)) * node_cardinality
        counts = np.bincount(combined_index, minlength=total_states).astype(float)
        return counts.reshape(tuple(parent_cardinalities) + (node_cardinality,))

    def _fit_local_cpd(self, frame: pd.DataFrame, node: str, parents: List[str]) -> LocalDiscreteCPD:
        counts = self._counts(frame, node, parents)
        probs = (counts + self.laplace) / (
            counts.sum(axis=-1, keepdims=True) + self.laplace * counts.shape[-1]
        )
        return LocalDiscreteCPD(
            node=node,
            parents=parents,
            parent_cardinalities=[self.cardinalities_[parent] for parent in parents],
            probabilities=probs,
        )

    def _lookup_probabilities(
        self,
        cpd: LocalDiscreteCPD,
        row_values: Dict[str, np.ndarray],
        node_values: np.ndarray,
        target_values: np.ndarray | None = None,
    ) -> np.ndarray:
        if not cpd.parents:
            return cpd.probabilities[node_values]
        parent_index = self._parent_index(cpd, row_values, target_values=target_values)
        flat_probabilities = cpd.probabilities.reshape(-1, cpd.probabilities.shape[-1])
        return flat_probabilities[parent_index, node_values]

    def _target_probabilities(
        self,
        cpd: LocalDiscreteCPD,
        row_values: Dict[str, np.ndarray],
        classes: np.ndarray,
        n_rows: int,
    ) -> np.ndarray:
        if not cpd.parents:
            base = cpd.probabilities[classes]
            return np.broadcast_to(base, (n_rows, len(classes)))
        parent_index = self._parent_index(cpd, row_values)
        flat_probabilities = cpd.probabilities.reshape(-1, cpd.probabilities.shape[-1])
        return flat_probabilities[parent_index[:, None], classes[None, :]]

    def _parent_index(
        self,
        cpd: LocalDiscreteCPD,
        row_values: Dict[str, np.ndarray],
        target_values: np.ndarray | None = None,
    ) -> np.ndarray:
        parent_values = tuple(
            target_values
            if parent == self.target_column and target_values is not None
            else row_values[parent]
            for parent in cpd.parents
        )
        return np.ravel_multi_index(parent_values, dims=tuple(cpd.parent_cardinalities))


def naive_bayes_graph(columns: Iterable[str], target_column: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for column in columns:
        if column != target_column:
            graph.add_edge(target_column, column)
    return graph


def tan_graph(frame: pd.DataFrame, target_column: str) -> nx.DiGraph:
    features = [column for column in frame.columns if column != target_column]
    graph = naive_bayes_graph(frame.columns, target_column)
    if len(features) <= 1:
        return graph

    weights: Dict[tuple[str, str], float] = {}
    for left_idx, left in enumerate(features):
        for right in features[left_idx + 1 :]:
            weights[(left, right)] = conditional_mutual_information(
                frame[left].to_numpy(dtype=int),
                frame[right].to_numpy(dtype=int),
                frame[target_column].to_numpy(dtype=int),
            )
    tree = nx.Graph()
    tree.add_nodes_from(features)
    for (left, right), weight in weights.items():
        tree.add_edge(left, right, weight=weight)
    spanning = nx.maximum_spanning_tree(tree)
    root = features[0]
    for parent, child in nx.bfs_edges(spanning, root):
        graph.add_edge(parent, child)
    return graph


def kdb_graph(frame: pd.DataFrame, target_column: str, k: int = 2) -> nx.DiGraph:
    y = frame[target_column].to_numpy(dtype=int)
    features = [column for column in frame.columns if column != target_column]
    ranking = sorted(
        features,
        key=lambda feature: mutual_information(frame[feature].to_numpy(dtype=int), y),
        reverse=True,
    )
    graph = naive_bayes_graph(frame.columns, target_column)
    for idx, feature in enumerate(ranking):
        previous = ranking[:idx]
        if not previous:
            continue
        scored = sorted(
            previous,
            key=lambda parent: conditional_mutual_information(
                frame[feature].to_numpy(dtype=int),
                frame[parent].to_numpy(dtype=int),
                y,
            ),
            reverse=True,
        )
        for parent in scored[:k]:
            graph.add_edge(parent, feature)
    return graph


class AODEClassifier:
    """Average one-dependence estimator for discrete tables."""

    def __init__(self, target_column: str, laplace: float = 1.0) -> None:
        self.target_column = target_column
        self.laplace = laplace
        self.cardinalities_: Dict[str, int] = {}
        self.columns_: List[str] = []
        self.parents_: List[str] = []
        self.joint_counts_: Dict[str, np.ndarray] = {}
        self.child_counts_: Dict[tuple[str, str], np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "AODEClassifier":
        self.columns_ = list(frame.columns)
        self.parents_ = [column for column in self.columns_ if column != self.target_column]
        self.cardinalities_ = {column: int(frame[column].max()) + 1 for column in self.columns_}
        y_card = self.cardinalities_[self.target_column]

        for parent in self.parents_:
            parent_card = self.cardinalities_[parent]
            joint = np.zeros((y_card, parent_card), dtype=float)
            for y_value, p_value in zip(
                frame[self.target_column].to_numpy(dtype=int),
                frame[parent].to_numpy(dtype=int),
            ):
                joint[y_value, p_value] += 1
            self.joint_counts_[parent] = joint
            for child in self.parents_:
                if child == parent:
                    continue
                child_card = self.cardinalities_[child]
                counts = np.zeros((y_card, parent_card, child_card), dtype=float)
                for y_value, p_value, c_value in zip(
                    frame[self.target_column].to_numpy(dtype=int),
                    frame[parent].to_numpy(dtype=int),
                    frame[child].to_numpy(dtype=int),
                ):
                    counts[y_value, p_value, c_value] += 1
                self.child_counts_[(parent, child)] = counts
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        y_card = self.cardinalities_[self.target_column]
        probabilities = np.zeros((len(frame), y_card), dtype=float)
        for row_idx, row in enumerate(frame.itertuples(index=False, name=None)):
            values = dict(zip(frame.columns, row))
            scores = np.zeros(y_card, dtype=float)
            for parent in self.parents_:
                parent_value = int(values[parent])
                joint = self.joint_counts_[parent]
                parent_term = joint[:, parent_value] + self.laplace
                parent_term /= joint.sum() + self.laplace * joint.size
                component = np.log(parent_term + 1e-12)
                for child in self.parents_:
                    if child == parent:
                        continue
                    child_value = int(values[child])
                    counts = self.child_counts_[(parent, child)]
                    numerator = counts[:, parent_value, child_value] + self.laplace
                    denominator = counts[:, parent_value, :].sum(axis=1) + self.laplace * counts.shape[-1]
                    component += np.log(numerator / denominator + 1e-12)
                scores += np.exp(component - component.max())
            probabilities[row_idx] = scores / scores.sum()
        return probabilities


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x_card = int(x.max()) + 1
    y_card = int(y.max()) + 1
    joint = np.zeros((x_card, y_card), dtype=float)
    for x_val, y_val in zip(x, y):
        joint[int(x_val), int(y_val)] += 1
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (px @ py)[mask])))


def conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    score = 0.0
    for z_value in range(int(z.max()) + 1):
        mask = z == z_value
        if mask.sum() == 0:
            continue
        weight = mask.mean()
        score += weight * mutual_information(x[mask], y[mask])
    return float(score)
