from __future__ import annotations

import numpy as np
import pandas as pd

from neural_bn.baselines import (
    AODEClassifier,
    KDBClassifier,
    NaiveBayesBNClassifier,
    TANClassifier,
    discover_tabpfn_checkpoint,
    external_baseline_availability,
)
from neural_bn.config import PipelineConfig
from neural_bn.data import TabularPreprocessor
from neural_bn.synthetic import make_mixed_classification_dataset


def test_builtin_bn_baselines_fit_and_predict():
    dataset = make_mixed_classification_dataset(n_samples=120, n_features=4, random_state=9)
    preprocessor = TabularPreprocessor(target_column=dataset.target_column)
    discrete = preprocessor.fit_transform(dataset.frame).discrete_frame
    features = discrete.drop(columns=[dataset.target_column])

    nb = NaiveBayesBNClassifier(dataset.target_column).fit(discrete)
    tan = TANClassifier(dataset.target_column).fit(discrete)
    kdb = KDBClassifier(dataset.target_column, k=2).fit(discrete)
    aode = AODEClassifier(dataset.target_column).fit(discrete)

    for estimator in (nb, tan, kdb, aode):
        probabilities = estimator.predict_proba(features)
        assert probabilities.shape[0] == len(features)
        assert probabilities.shape[1] == 2


def test_external_baseline_registry_reports_expected_keys():
    availability = external_baseline_availability()
    assert {"xgboost", "lightgbm", "catboost", "tabpfn", "tabm", "ft_transformer", "tabicl"} <= set(
        availability
    )


def test_discover_tabpfn_checkpoint_prefers_explicit_env_path(tmp_path, monkeypatch):
    checkpoint = tmp_path / "explicit.ckpt"
    checkpoint.write_text("checkpoint")
    monkeypatch.setenv("NEURAL_BN_TABPFN_MODEL_PATH", str(checkpoint))
    monkeypatch.delenv("NEURAL_BN_TABPFN_CACHE_DIR", raising=False)

    assert discover_tabpfn_checkpoint() == checkpoint


def test_tabular_preprocessor_maps_missing_categorical_values_to_valid_targets():
    frame = pd.DataFrame(
        {
            "feature_a": ["left", np.nan, "right", np.nan],
            "feature_b": ["cold", "warm", np.nan, "cold"],
            "label": ["yes", "no", "yes", "no"],
        }
    )

    prepared = TabularPreprocessor(target_column="label").fit_transform(frame)

    for column in prepared.columns:
        if column.kind != "categorical":
            continue
        encoded = prepared.target_arrays[column.name]
        assert int(encoded.min()) >= 0
        assert int(encoded.max()) < column.target_size
