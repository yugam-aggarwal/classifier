from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from neural_bn.baselines import BaselineAvailability
from neural_bn.datasets import DatasetArtifact, InMemoryDatasetAdapter
from neural_bn.experiments import (
    EvidenceStageSpec,
    _encode_supervised_features,
    _instantiate_external_model,
    ablation_registry,
    build_baseline_runners,
    evidence_stage_registry,
    run_benchmark_suite,
    run_staged_evidence_pass,
    summarize_fast_mode_design_probe,
    summarize_paper_positioning,
    summarize_paper_results,
)
from neural_bn.config import PipelineConfig
from neural_bn.synthetic import make_mixed_classification_dataset


def _small_config(target_column: str) -> PipelineConfig:
    config = PipelineConfig(target_column=target_column)
    config.screener.epochs = 2
    config.screener.bootstrap_rounds = 1
    config.screener.candidate_pool_size = 3
    config.search.max_iters = 5
    return config


def _single_split():
    dataset = make_mixed_classification_dataset(n_samples=120, n_features=4, random_state=17)
    artifact = DatasetArtifact(
        name="tiny_structural",
        regime="known_graph",
        frame=dataset.frame,
        target_column=dataset.target_column,
        truth_graph=dataset.graph,
    )
    return next(InMemoryDatasetAdapter(artifact, repeats=1).iter_splits(11))


def test_experiment_result_serialization_includes_metadata_and_artifacts():
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["neural_screened_bn"]
    result = runner.run(split, config, ablation_registry()["default"])
    payload = result.to_dict()

    assert payload["metadata"]["dataset_name"] == split.artifact.name
    assert payload["metadata"]["model_name"] == "neural_screened_bn"
    assert "split_metadata" in payload
    assert "candidate_pools" in payload
    assert "graph_edges" in payload
    assert "selected_start" in payload["search_stats"]
    assert "candidate_starts" in payload["search_stats"]
    assert "start_summaries" in payload["search_stats"]


def test_ablation_no_screener_expands_candidate_pools():
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["neural_screened_bn"]

    default_result = runner.run(split, config, ablation_registry()["default"])
    no_screener_result = runner.run(split, config, ablation_registry()["no_screener"])

    assert default_result.structural["candidate_pool_reduction"] >= 0.0
    assert no_screener_result.structural["candidate_pool_reduction"] == 0.0
    target = split.artifact.target_column
    assert len(no_screener_result.candidate_pools[target]) == len(split.artifact.frame.columns) - 1


def test_baseline_runner_consistency_for_builtin_and_search_methods():
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runners = build_baseline_runners()

    for name in ("naive_bayes", "tan", "greedy_hc_bn"):
        result = runners[name].run(split, config, ablation_registry()["default"])
        assert result.status == "ok"
        assert result.metadata.model_name == name
        assert "log_loss" in result.predictive


def test_smoke_suite_runs_and_produces_tables(tmp_path):
    config = _small_config("label")
    output_path = tmp_path / "smoke_results.json"
    suite = run_benchmark_suite(
        suite_name="smoke",
        base_config=config,
        dataset_names=["synthetic_tiny", "sklearn_iris"],
        baseline_names=["neural_screened_bn", "naive_bayes"],
        seeds=[5],
        ablations=["default"],
        output_path=output_path,
    )

    assert suite.records
    assert "predictive_leaderboard" in suite.tables
    assert any(record.status == "ok" for record in suite.records)
    assert "run_metadata" in suite.to_dict()
    assert suite.run_metadata["suite_name"] == "smoke"
    assert suite.run_metadata["output_path"] == str(output_path)
    assert (output_path.parent / ".runtime" / "mplconfig").is_dir()


def test_encode_supervised_features_returns_aligned_dataframes():
    train_frame = pd.DataFrame(
        {
            "color": ["red", "[blue]", "red"],
            "value": [1.0, 2.0, 3.0],
            "label": ["yes", "no", "yes"],
        }
    )
    test_frame = pd.DataFrame(
        {
            "color": ["<green>", "[blue]"],
            "value": [4.0, 5.0],
            "label": ["no", "yes"],
        }
    )

    x_train, x_test, y_train, y_test = _encode_supervised_features(train_frame, test_frame, "label")

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert list(x_train.columns) == list(x_test.columns)
    assert all("[" not in column and "]" not in column and "<" not in column for column in x_train.columns)
    assert y_train.shape == (3,)
    assert y_test.shape == (2,)


def test_builtin_bn_runner_records_runtime_breakdown():
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["naive_bayes"]

    result = runner.run(split, config, ablation_registry()["default"])

    assert result.runtimes["total_seconds"] > 0.0
    assert result.runtimes["preprocess_seconds"] >= 0.0
    assert result.runtimes["fit_seconds"] >= 0.0
    assert result.runtimes["predict_seconds"] >= 0.0


def test_external_tree_runner_preserves_feature_names_and_records_runtime(monkeypatch):
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["xgboost"]

    class DummyTreeModel:
        def fit(self, x, y):
            self.fit_x = x
            self.fit_y = y
            return self

        def predict_proba(self, x):
            self.predict_x = x
            return np.tile(np.array([[0.4, 0.6]]), (len(x), 1))

    dummy = DummyTreeModel()
    monkeypatch.setattr(
        "neural_bn.experiments.external_baseline_availability",
        lambda: {"xgboost": BaselineAvailability(name="xgboost", available=True, detail="installed")},
    )
    monkeypatch.setattr("neural_bn.experiments._instantiate_external_model", lambda *args, **kwargs: dummy)

    result = runner.run(split, config, ablation_registry()["default"])

    assert isinstance(dummy.fit_x, pd.DataFrame)
    assert isinstance(dummy.predict_x, pd.DataFrame)
    assert list(dummy.fit_x.columns) == list(dummy.predict_x.columns)
    assert result.runtimes["total_seconds"] > 0.0
    assert result.runtimes["encode_seconds"] >= 0.0


def test_tabpfn_runner_uses_numpy_and_records_runtime(monkeypatch):
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["tabpfn"]

    class DummyTabPFNModel:
        def fit(self, x, y):
            self.fit_x = x
            self.fit_y = y
            return self

        def predict_proba(self, x):
            self.predict_x = x
            return np.tile(np.array([[0.45, 0.55]]), (len(x), 1))

    dummy = DummyTabPFNModel()
    monkeypatch.setattr(
        "neural_bn.experiments.external_baseline_availability",
        lambda: {"tabpfn": BaselineAvailability(name="tabpfn", available=True, detail="installed")},
    )
    monkeypatch.setattr("neural_bn.experiments._instantiate_tabpfn_model", lambda *args, **kwargs: dummy)
    monkeypatch.setattr("neural_bn.experiments._tabpfn_quiet_environment", lambda: nullcontext())

    result = runner.run(split, config, ablation_registry()["default"])

    assert isinstance(dummy.fit_x, np.ndarray)
    assert isinstance(dummy.predict_x, np.ndarray)
    assert result.runtimes["total_seconds"] > 0.0
    assert result.runtimes["encode_seconds"] >= 0.0


def test_instantiate_external_model_uses_gpu_env_for_xgboost(monkeypatch):
    captured: dict[str, object] = {}

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeModule:
        XGBClassifier = FakeXGBClassifier

    monkeypatch.setenv("NEURAL_BN_EXTERNAL_DEVICE", "cuda:1")
    monkeypatch.setattr("neural_bn.experiments.importlib.import_module", lambda name: FakeModule())

    _instantiate_external_model("xgboost", seed=5, n_classes=2)

    assert captured["device"] == "cuda:1"
    assert captured["tree_method"] == "hist"


def test_instantiate_external_model_uses_gpu_env_for_lightgbm(monkeypatch):
    captured: dict[str, object] = {}

    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeModule:
        LGBMClassifier = FakeLGBMClassifier

    monkeypatch.setenv("NEURAL_BN_LIGHTGBM_DEVICE", "cuda:1")
    monkeypatch.setattr("neural_bn.experiments.importlib.import_module", lambda name: FakeModule())

    _instantiate_external_model("lightgbm", seed=5, n_classes=2)

    assert captured["device_type"] == "gpu"
    assert captured["gpu_device_id"] == 1
    assert captured["n_estimators"] == 64
    assert captured["force_col_wise"] is True
    assert captured["n_jobs"] == 1


def test_suite_progress_emits_stderr(capsys):
    config = _small_config("label")
    _ = run_benchmark_suite(
        suite_name="smoke",
        base_config=config,
        dataset_names=["synthetic_tiny"],
        baseline_names=["neural_screened_bn"],
        seeds=[5],
        ablations=["default"],
        progress=True,
    )

    captured = capsys.readouterr()
    assert "[suite:smoke]" in captured.err
    assert "[screener]" in captured.err
    assert "[search]" in captured.err
    assert "[pipeline]" in captured.err


def test_leaderboards_only_use_default_ablation_runs():
    config = _small_config("label")
    suite = run_benchmark_suite(
        suite_name="smoke",
        base_config=config,
        dataset_names=["synthetic_tiny"],
        baseline_names=["neural_screened_bn"],
        seeds=[5],
        ablations=["default", "no_screener"],
    )

    row = suite.tables["predictive_leaderboard"][0]
    assert row["model_name"] == "neural_screened_bn"
    assert row["num_runs"] == 1
    assert any(
        item["dataset_name"] == "synthetic_tiny" and item["ablation_label"] == "no_screener"
        for item in suite.tables["ablation_summary"]
    )


def test_known_graph_fast_ablation_only_applies_on_known_graph():
    base_config = _small_config("label")
    ablation = ablation_registry()["known_graph_fast"]

    known_graph_config = ablation.apply(base_config, dataset_regime="known_graph", num_columns=20)
    real_config = ablation.apply(base_config, dataset_regime="real", num_columns=20)

    assert known_graph_config.screener.bootstrap_rounds == 1
    assert known_graph_config.screener.epochs == 15
    assert known_graph_config.screener.candidate_pool_size == 3
    assert known_graph_config.screener.use_interaction_bundles is True
    assert known_graph_config.search.warm_start == "naive_bayes"
    assert known_graph_config.search.multi_start_candidates == ("naive_bayes",)
    assert known_graph_config.search.max_iters == 12
    assert known_graph_config.search.validation_prune_passes == 0

    assert real_config.screener.bootstrap_rounds == base_config.screener.bootstrap_rounds
    assert real_config.screener.epochs == base_config.screener.epochs
    assert real_config.search.warm_start == base_config.search.warm_start
    assert real_config.search.max_iters == base_config.search.max_iters
    assert real_config.search.validation_prune_passes == base_config.search.validation_prune_passes


def test_known_graph_fast_ablation_uses_large_iter_cap_for_wide_problem():
    config = _small_config("label")
    ablation = ablation_registry()["known_graph_fast"]

    updated = ablation.apply(config, dataset_regime="known_graph", num_columns=21)

    assert updated.search.max_iters == 20


@pytest.mark.parametrize(
    "ablation_name,expected_strategy",
    [
        ("mi_filter_fast", "mi_filter"),
        ("single_pass_neural_fast", "neural"),
        ("hybrid_mi_neural_fast", "hybrid_mi_neural"),
    ],
)
def test_fast_mode_ablations_only_apply_on_known_graph(ablation_name, expected_strategy):
    base_config = _small_config("label")
    ablation = ablation_registry()[ablation_name]

    known_graph_config = ablation.apply(base_config, dataset_regime="known_graph", num_columns=20)
    real_config = ablation.apply(base_config, dataset_regime="real", num_columns=20)

    assert known_graph_config.screener.strategy == expected_strategy
    assert known_graph_config.screener.epochs == 15
    assert known_graph_config.screener.candidate_pool_size == 3
    assert known_graph_config.screener.candidate_pool_soft_cap == 3
    assert real_config.screener.strategy == base_config.screener.strategy


@pytest.mark.parametrize(
    "ablation_name",
    ["mi_filter_fast", "single_pass_neural_fast", "hybrid_mi_neural_fast"],
)
def test_fast_mode_ablations_run_end_to_end(ablation_name):
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["neural_screened_bn"]

    result = runner.run(split, config, ablation_registry()[ablation_name])

    assert result.status == "ok"
    assert result.runtimes["total_seconds"] > 0.0
    assert "predicted_edge_count" in result.structural
    assert result.screener_diagnostics["screening_strategy"] in {"mi_filter", "neural", "hybrid_mi_neural"}


def test_tabpfn_runner_returns_schema_compatible_result():
    split = _single_split()
    config = _small_config(split.artifact.target_column)
    runner = build_baseline_runners()["tabpfn"]
    result = runner.run(split, config, ablation_registry()["default"])
    payload = result.to_dict()

    assert payload["metadata"]["model_name"] == "tabpfn"
    assert result.status in {"ok", "skipped"}
    assert "notes" in payload
    assert "split_metadata" in payload


def test_paper_minimum_slice_runs_with_structural_fixture_and_real_dataset(tmp_path):
    expert_root = tmp_path / "datasets" / "expert_bn"
    for name in ("alarm", "child", "insurance"):
        path = expert_root / name
        path.mkdir(parents=True)
        pd.DataFrame(
            {
                "x0": [0, 1, 0, 1, 0, 1, 0, 1],
                "x1": [1, 0, 1, 0, 1, 0, 1, 0],
                "label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
            }
        ).to_csv(path / "frame.csv", index=False)
        pd.DataFrame(
            {
                "source": ["x0", "x1"],
                "target": ["label", "label"],
            }
        ).to_csv(path / "edges.csv", index=False)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "target_column": "label",
                    "source": f"fixture::{name}",
                    "notes": [f"{name} fixture"],
                }
            )
        )

    config = _small_config("label")
    suite = run_benchmark_suite(
        suite_name="paper_minimum",
        base_config=config,
        base_dir=tmp_path,
        dataset_names=["alarm", "sklearn_iris"],
        baseline_names=["neural_screened_bn", "naive_bayes", "tabpfn"],
        seeds=[5],
        ablations=["default"],
    )

    assert suite.records
    assert "predictive_leaderboard" in suite.tables
    assert "structural_leaderboard" in suite.tables
    assert "search_complexity" in suite.tables
    assert "skipped_items" in suite.tables
    assert any(
        record.metadata.dataset_name == "alarm"
        and record.status == "ok"
        and record.structural is not None
        and "shd" in record.structural
        for record in suite.records
    )
    assert all(record.metadata.dataset_name for record in suite.records if record.status == "ok")


def test_run_staged_evidence_pass_writes_artifacts_and_summary(tmp_path):
    config = _small_config("label")
    stage_specs = {
        "sanity_default": EvidenceStageSpec(
            name="sanity_default",
            suite_name="smoke",
            output_filename="sanity_default.json",
            dataset_names=["synthetic_tiny", "sklearn_iris"],
            baseline_names=["neural_screened_bn", "naive_bayes"],
            seeds=[5],
            ablations=["default"],
        ),
        "paper_minimum_full": EvidenceStageSpec(
            name="paper_minimum_full",
            suite_name="smoke",
            output_filename="paper_minimum_full.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default", "no_screener"],
        ),
    }

    summary = run_staged_evidence_pass(
        base_config=config,
        artifact_dir=tmp_path,
        stage_specs=stage_specs,
        stages=["sanity_default", "paper_minimum_full"],
        screener_device="cpu",
        tabpfn_device="cpu",
    )

    assert (tmp_path / "sanity_default.json").exists()
    assert (tmp_path / "paper_minimum_full.json").exists()
    assert (tmp_path / "evidence_summary.json").exists()
    assert (tmp_path / ".runtime" / "mplconfig").is_dir()
    assert summary["completed_stages"] == ["sanity_default", "paper_minimum_full"]
    assert "sanity_default" in summary["stage_tables"]
    assert summary["run_metadata"]["artifact_dir"] == str(tmp_path)
    assert summary["run_metadata"]["requested_screener_device"] == "cpu"
    assert summary["run_metadata"]["requested_tabpfn_device"] == "cpu"
    assert "decision" in summary
    assert summary["decision"]["basis_stage"] == "paper_minimum_full"
    assert summary["decision"]["verdict"] in {"go", "no_go"}


def test_run_staged_evidence_pass_emits_progress(tmp_path, capsys):
    config = _small_config("label")
    stage_specs = {
        "sanity_default": EvidenceStageSpec(
            name="sanity_default",
            suite_name="smoke",
            output_filename="sanity_default.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
    }

    _ = run_staged_evidence_pass(
        base_config=config,
        artifact_dir=tmp_path,
        stage_specs=stage_specs,
        stages=["sanity_default"],
        screener_device="cpu",
        tabpfn_device="cpu",
        progress=True,
    )

    captured = capsys.readouterr()
    assert "[evidence]" in captured.err
    assert "[stage:sanity_default]" in captured.err


def test_known_graph_fast_probe_stage_writes_ablation_runtime_rows(tmp_path):
    config = _small_config("label")
    stage_specs = {
        "sanity_default": EvidenceStageSpec(
            name="sanity_default",
            suite_name="smoke",
            output_filename="sanity_default.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "structural_default": EvidenceStageSpec(
            name="structural_default",
            suite_name="smoke",
            output_filename="structural_default.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "paper_minimum_full": EvidenceStageSpec(
            name="paper_minimum_full",
            suite_name="smoke",
            output_filename="paper_minimum_full.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "known_graph_fast_probe": EvidenceStageSpec(
            name="known_graph_fast_probe",
            suite_name="smoke",
            output_filename="known_graph_fast_probe.json",
            dataset_names=["synthetic_tiny", "sklearn_iris"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default", "known_graph_fast"],
        ),
    }

    summary = run_staged_evidence_pass(
        base_config=config,
        artifact_dir=tmp_path,
        stage_specs=stage_specs,
        stages=["known_graph_fast_probe"],
        screener_device="cpu",
        tabpfn_device="cpu",
    )

    artifact_path = tmp_path / "known_graph_fast_probe.json"
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text())
    assert any(
        row["ablation_label"] == "known_graph_fast" and "avg_total_seconds" in row
        for row in payload["tables"]["ablation_summary"]
    )
    assert summary["completed_stages"] == ["known_graph_fast_probe"]


def test_default_stage_selection_excludes_known_graph_fast_probe(tmp_path):
    config = _small_config("label")
    stage_specs = {
        "sanity_default": EvidenceStageSpec(
            name="sanity_default",
            suite_name="smoke",
            output_filename="sanity_default.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "structural_default": EvidenceStageSpec(
            name="structural_default",
            suite_name="smoke",
            output_filename="structural_default.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "paper_minimum_full": EvidenceStageSpec(
            name="paper_minimum_full",
            suite_name="smoke",
            output_filename="paper_minimum_full.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default"],
        ),
        "known_graph_fast_probe": EvidenceStageSpec(
            name="known_graph_fast_probe",
            suite_name="smoke",
            output_filename="known_graph_fast_probe.json",
            dataset_names=["synthetic_tiny"],
            baseline_names=["neural_screened_bn"],
            seeds=[5],
            ablations=["default", "known_graph_fast"],
        ),
    }

    summary = run_staged_evidence_pass(
        base_config=config,
        artifact_dir=tmp_path,
        stage_specs=stage_specs,
        screener_device="cpu",
        tabpfn_device="cpu",
    )

    assert summary["requested_stages"] == ["sanity_default", "structural_default", "paper_minimum_full"]
    assert not (tmp_path / "known_graph_fast_probe.json").exists()


def test_evidence_registry_includes_known_graph_fast_probe():
    registry = evidence_stage_registry()

    assert "known_graph_fast_probe" in registry
    assert registry["known_graph_fast_probe"].ablations == ["default", "known_graph_fast"]


def test_evidence_registry_includes_fast_mode_design_probe():
    registry = evidence_stage_registry()

    assert "fast_mode_design_probe" in registry
    assert registry["fast_mode_design_probe"].baseline_names == ["neural_screened_bn", "greedy_hc_bn"]
    assert registry["fast_mode_design_probe"].ablations == [
        "default",
        "known_graph_fast",
        "mi_filter_fast",
        "single_pass_neural_fast",
        "hybrid_mi_neural_fast",
    ]


def test_summarize_fast_mode_design_probe_reports_runtime_and_shd_for_each_candidate():
    datasets = ["alarm", "child", "insurance", "synthetic_medium", "synthetic_small"]
    records = []
    for ablation_name in ("known_graph_fast", "mi_filter_fast", "single_pass_neural_fast", "hybrid_mi_neural_fast"):
        for dataset_name in datasets:
            records.append(
                {
                    "metadata": {
                        "model_name": "neural_screened_bn",
                        "dataset_name": dataset_name,
                        "dataset_regime": "known_graph",
                        "ablation_label": ablation_name,
                    },
                    "status": "ok",
                }
            )
    payload = {
        "records": records,
        "tables": {
            "structural_leaderboard": [
                {"dataset_name": dataset_name, "model_name": "neural_screened_bn", "avg_shd": 10.0}
                for dataset_name in datasets
            ],
            "search_complexity": [
                {"dataset_name": dataset_name, "model_name": "neural_screened_bn", "avg_total_seconds": 10.0}
                for dataset_name in datasets
            ]
            + [
                {"dataset_name": dataset_name, "model_name": "greedy_hc_bn", "avg_total_seconds": 4.0}
                for dataset_name in datasets
            ],
            "ablation_summary": [
                {
                    "dataset_name": dataset_name,
                    "ablation_label": ablation_name,
                    "avg_shd": 11.0,
                    "avg_total_seconds": 3.0,
                }
                for ablation_name in (
                    "known_graph_fast",
                    "mi_filter_fast",
                    "single_pass_neural_fast",
                    "hybrid_mi_neural_fast",
                )
                for dataset_name in datasets
            ],
            "skipped_items": [],
        },
    }

    summary = summarize_fast_mode_design_probe(payload)

    for ablation_name in ("known_graph_fast", "mi_filter_fast", "single_pass_neural_fast", "hybrid_mi_neural_fast"):
        candidate = summary["candidates"][ablation_name]
        assert len(candidate["dataset_rows"]) == 5
        assert candidate["promotion_gate"]["skipped_runs"] == 0
        assert all("candidate_total_seconds" in row and "candidate_shd" in row for row in candidate["dataset_rows"])


def test_summarize_paper_positioning_reproduces_current_artifact_headlines(tmp_path):
    source_dir = Path("artifacts") / "benchmarks"
    artifact_dir = tmp_path / "benchmarks"
    artifact_dir.mkdir(parents=True)
    for name in ("evidence_summary.json", "paper_minimum_full.json", "known_graph_fast_probe.json", "fast_mode_design_probe.json"):
        shutil.copyfile(source_dir / name, artifact_dir / name)

    summary = summarize_paper_positioning(artifact_dir)

    assert summary["headline_verdicts"]["quality_first_default"]["supported"] is True
    assert summary["headline_verdicts"]["runtime_limitation"]["supported"] is True
    assert len(summary["explicit_bn_structural_comparison"]) == 5
    assert sum(row["shd_delta"] > 0 for row in summary["explicit_bn_structural_comparison"]) == 3
    assert sum(row["competitive"] for row in summary["strong_predictive_baseline_comparison"]) == 2
    assert sum(row["runtime_win"] for row in summary["runtime_vs_greedy_hc_comparison"]) == 0
    assert (artifact_dir / "paper_positioning.json").exists()
    assert (artifact_dir / "paper_positioning.md").exists()
    assert (artifact_dir / "paper_results.json").exists()
    assert (artifact_dir / "paper_results.md").exists()
    assert summary["paper_results_headlines"]["structural_win_count"] == 3


def test_summarize_paper_results_builds_tables_and_narrative(tmp_path):
    source_dir = Path("artifacts") / "benchmarks"
    artifact_dir = tmp_path / "benchmarks"
    artifact_dir.mkdir(parents=True)
    for name in ("evidence_summary.json", "paper_minimum_full.json", "fast_mode_design_probe.json"):
        shutil.copyfile(source_dir / name, artifact_dir / name)

    payload = summarize_paper_results(artifact_dir)

    assert payload["headline_metrics"]["structural_win_count"] == 3
    assert payload["headline_metrics"]["predictive_competitive_count"] == 2
    assert payload["headline_metrics"]["runtime_win_count_vs_greedy"] == 0
    assert len(payload["structural_table"]) == 5
    assert len(payload["predictive_table"]) == 2
    assert len(payload["runtime_table"]) == 5
    assert len(payload["results_narrative"]) == 3
    assert len(payload["limitations_narrative"]) == 2
    assert (artifact_dir / "paper_results.json").exists()
    assert (artifact_dir / "paper_results.md").exists()
