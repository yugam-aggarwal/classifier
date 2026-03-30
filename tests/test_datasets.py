from __future__ import annotations

import json
import pandas as pd

from neural_bn.datasets import LocalStructuralDatasetAdapter, SklearnClassificationDatasetAdapter
from neural_bn.expert_bn import materialize_expert_bn_assets


def test_sklearn_dataset_adapter_loads_real_classification_dataset():
    adapter = SklearnClassificationDatasetAdapter("sklearn_iris", repeats=1)
    artifact = adapter.load()

    assert artifact.name == "sklearn_iris"
    assert artifact.regime == "real"
    assert artifact.target_column in artifact.frame.columns
    assert len(artifact.frame) > 0


def test_local_structural_dataset_adapter_loads_graph_and_splits(tmp_path):
    root = tmp_path / "alarm"
    root.mkdir()
    pd.DataFrame(
        {
            "x0": [0, 1, 0, 1, 0, 1],
            "x1": [1, 0, 1, 0, 1, 0],
            "label": ["yes", "no", "yes", "no", "yes", "no"],
        }
    ).to_csv(root / "frame.csv", index=False)
    pd.DataFrame(
        {
            "source": ["x0", "x1"],
            "target": ["label", "label"],
        }
    ).to_csv(root / "edges.csv", index=False)

    adapter = LocalStructuralDatasetAdapter(root, repeats=1)
    artifact = adapter.load()
    split = next(adapter.iter_splits(7))

    assert artifact.truth_graph is not None
    assert artifact.truth_graph.has_edge("x0", "label")
    assert split.artifact.name == "alarm"
    assert len(split.train_frame) > 0
    assert len(split.test_frame) > 0


def test_local_structural_dataset_adapter_reads_optional_metadata(tmp_path):
    root = tmp_path / "insurance"
    root.mkdir()
    pd.DataFrame(
        {
            "feature": [0, 1, 0, 1, 1, 0],
            "label": ["yes", "no", "yes", "no", "yes", "no"],
        }
    ).to_csv(root / "frame.csv", index=False)
    pd.DataFrame({"source": ["feature"], "target": ["label"]}).to_csv(root / "edges.csv", index=False)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "target_column": "label",
                "source": "local_fixture::insurance",
                "notes": ["fixture dataset"],
            }
        )
    )

    artifact = LocalStructuralDatasetAdapter(root, repeats=1).load()

    assert artifact.source == "local_fixture::insurance"
    assert artifact.notes == ["fixture dataset"]


def test_materialize_expert_bn_assets_writes_canonical_contract(tmp_path):
    summary = materialize_expert_bn_assets(tmp_path, dataset_names=["alarm"], overwrite=True)

    assert summary["alarm"]["status"] == "written"
    frame = pd.read_csv(tmp_path / "alarm" / "frame.csv")
    edges = pd.read_csv(tmp_path / "alarm" / "edges.csv")
    metadata = json.loads((tmp_path / "alarm" / "metadata.json").read_text())

    assert metadata["source"] == "pgmpy::example_model::alarm"
    assert metadata["target_column"] == "PRESS"
    assert "PRESS" in frame.columns
    assert {"source", "target"} <= set(edges.columns)
    assert len(frame) == 2500
