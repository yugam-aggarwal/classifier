"""Preprocessing for neural screening and discrete BN search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass(slots=True)
class ColumnInfo:
    name: str
    kind: str
    input_indices: List[int]
    target_size: int
    categories: List[str] | None = None
    mean: float | None = None
    std: float | None = None
    thresholds: List[float] | None = None


@dataclass(slots=True)
class PreparedData:
    encoded_matrix: np.ndarray
    target_arrays: Dict[str, np.ndarray]
    columns: List[ColumnInfo]
    discrete_frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str


class TabularPreprocessor:
    """Creates both neural inputs and discrete BN-search tables."""

    def __init__(
        self,
        target_column: str,
        n_bins: int = 5,
        min_samples_per_bin: int = 20,
        categorical_columns: Iterable[str] | None = None,
    ) -> None:
        self.target_column = target_column
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.explicit_categorical = set(categorical_columns or [])
        self.columns_: List[ColumnInfo] = []
        self.feature_columns_: List[str] = []
        self.encoder_: OneHotEncoder | None = None
        self.categorical_columns_: List[str] = []
        self.continuous_columns_: List[str] = []
        self._fitted = False

    def fit_transform(self, frame: pd.DataFrame) -> PreparedData:
        self.fit(frame)
        return self.transform(frame)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        if self.target_column not in frame.columns:
            raise KeyError(f"Target column '{self.target_column}' not present.")

        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = []
        self.continuous_columns_ = []
        for column in self.feature_columns_:
            series = frame[column]
            if column == self.target_column:
                self.categorical_columns_.append(column)
            elif (
                column in self.explicit_categorical
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)
            ):
                self.categorical_columns_.append(column)
            else:
                self.continuous_columns_.append(column)

        categorical_frame = self._stringify_categorical_frame(frame[self.categorical_columns_])
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(categorical_frame)

        encoded_feature_names = self.encoder_.get_feature_names_out(self.categorical_columns_)
        start = 0
        self.columns_ = []
        for column in self.feature_columns_:
            if column in self.categorical_columns_:
                categories = [str(item) for item in self.encoder_.categories_[self.categorical_columns_.index(column)]]
                width = len(categories)
                info = ColumnInfo(
                    name=column,
                    kind="categorical",
                    input_indices=list(range(start, start + width)),
                    target_size=width,
                    categories=categories,
                )
                self.columns_.append(info)
                start += width
            else:
                numeric = pd.to_numeric(frame[column], errors="coerce").fillna(frame[column].median())
                mean = float(numeric.mean())
                std = float(numeric.std(ddof=0))
                if std < 1e-8:
                    std = 1.0
                thresholds = self._fit_supervised_bins(
                    numeric.to_numpy(),
                    frame[self.target_column].astype(str).to_numpy(),
                )
                info = ColumnInfo(
                    name=column,
                    kind="continuous",
                    input_indices=[start],
                    target_size=1,
                    mean=mean,
                    std=std,
                    thresholds=thresholds,
                )
                self.columns_.append(info)
                start += 1

        _ = encoded_feature_names
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> PreparedData:
        if not self._fitted or self.encoder_ is None:
            raise RuntimeError("The preprocessor must be fit before transform.")

        working = frame.copy()
        if self.target_column not in working.columns:
            target_info = next(column for column in self.columns_ if column.name == self.target_column)
            default_target = (target_info.categories or ["0"])[0]
            working[self.target_column] = default_target

        categorical_values = self._stringify_categorical_frame(working[self.categorical_columns_])
        categorical_encoded = self.encoder_.transform(categorical_values)
        blocks: List[np.ndarray] = []
        target_arrays: Dict[str, np.ndarray] = {}
        discrete_columns: Dict[str, np.ndarray] = {}
        cat_offset = 0
        for info in self.columns_:
            if info.kind == "categorical":
                block = categorical_encoded[:, cat_offset : cat_offset + info.target_size]
                blocks.append(block.astype(np.float32))
                values = categorical_values[info.name]
                mapping = {category: idx for idx, category in enumerate(info.categories or [])}
                target_arrays[info.name] = values.map(mapping).to_numpy(dtype=np.int64)
                discrete_columns[info.name] = target_arrays[info.name]
                cat_offset += info.target_size
            else:
                numeric = pd.to_numeric(working[info.name], errors="coerce").fillna(info.mean)
                standardized = ((numeric - float(info.mean)) / float(info.std)).to_numpy(dtype=np.float32)
                blocks.append(standardized.reshape(-1, 1))
                target_arrays[info.name] = standardized.astype(np.float32)
                discrete_columns[info.name] = self._digitize_with_thresholds(
                    numeric.to_numpy(dtype=float),
                    info.thresholds or [],
                )

        discrete_frame = pd.DataFrame(discrete_columns, index=frame.index)
        return PreparedData(
            encoded_matrix=np.concatenate(blocks, axis=1),
            target_arrays=target_arrays,
            columns=self.columns_,
            discrete_frame=discrete_frame,
            feature_columns=self.feature_columns_,
            target_column=self.target_column,
        )

    def transform_for_prediction(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self.transform(frame)
        return prepared.discrete_frame

    def _fit_supervised_bins(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        if len(np.unique(x)) <= self.n_bins:
            unique = np.unique(x)
            if len(unique) <= 1:
                return []
            return [float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:])]

        min_leaf = max(2, min(self.min_samples_per_bin, max(2, len(x) // max(self.n_bins, 2))))
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_leaf_nodes=self.n_bins,
            min_samples_leaf=min_leaf,
            random_state=0,
        )
        tree.fit(x.reshape(-1, 1), y)
        thresholds = tree.tree_.threshold
        good = sorted(float(value) for value in thresholds if value > -2.0)
        deduped: List[float] = []
        for value in good:
            if not deduped or abs(value - deduped[-1]) > 1e-9:
                deduped.append(value)
        return deduped

    @staticmethod
    def _digitize_with_thresholds(values: np.ndarray, thresholds: List[float]) -> np.ndarray:
        if not thresholds:
            return np.zeros(len(values), dtype=np.int64)
        return np.digitize(values, bins=np.asarray(thresholds), right=False).astype(np.int64)

    @staticmethod
    def _stringify_categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(TabularPreprocessor._stringify_categorical_series)

    @staticmethod
    def _stringify_categorical_series(series: pd.Series) -> pd.Series:
        normalized = series.where(~series.isna(), "nan")
        return normalized.astype(str)
