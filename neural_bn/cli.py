"""Command-line entry points for smoke tests and local runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import external_baseline_availability
from .config import PipelineConfig
from .expert_bn import materialize_expert_bn_assets
from .experiments import run_benchmark_suite, run_staged_evidence_pass, summarize_paper_positioning
from .pipeline import NeuralScreenedBNClassifier
from .synthetic import make_mixed_classification_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural-screened explicit BN classifiers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a synthetic smoke test")
    smoke.add_argument("--samples", type=int, default=400)
    smoke.add_argument("--features", type=int, default=6)
    smoke.add_argument("--screener-device")
    smoke.add_argument("--progress", action="store_true")

    train = subparsers.add_parser("train", help="Train on a CSV file")
    train.add_argument("csv_path", type=Path)
    train.add_argument("--target", required=True)
    train.add_argument("--screener-device")
    train.add_argument("--progress", action="store_true")

    run_suite = subparsers.add_parser("run-suite", help="Run a named benchmark suite")
    run_suite.add_argument("suite_name")
    run_suite.add_argument("--output", type=Path)
    run_suite.add_argument("--base-dir", type=Path, default=Path("."))
    run_suite.add_argument("--datasets", nargs="*")
    run_suite.add_argument("--baselines", nargs="*")
    run_suite.add_argument("--seeds", nargs="*", type=int)
    run_suite.add_argument("--ablations", nargs="*")
    run_suite.add_argument("--screener-device")
    run_suite.add_argument("--tabpfn-device")
    run_suite.add_argument("--no-progress", action="store_true")

    run_evidence = subparsers.add_parser("run-evidence", help="Run the staged evidence benchmark workflow")
    run_evidence.add_argument("--artifact-dir", type=Path, default=Path("artifacts") / "benchmarks")
    run_evidence.add_argument("--base-dir", type=Path, default=Path("."))
    run_evidence.add_argument("--stages", nargs="*")
    run_evidence.add_argument("--screener-device")
    run_evidence.add_argument("--tabpfn-device")
    run_evidence.add_argument("--no-progress", action="store_true")

    summarize_paper = subparsers.add_parser(
        "summarize-paper",
        help="Summarize frozen benchmark artifacts into paper-facing positioning outputs",
    )
    summarize_paper.add_argument("--artifact-dir", type=Path, default=Path("artifacts") / "benchmarks")

    materialize = subparsers.add_parser(
        "materialize-expert-bn",
        help="Generate canonical expert-BN assets under datasets/expert_bn",
    )
    materialize.add_argument("--root", type=Path, default=Path("datasets") / "expert_bn")
    materialize.add_argument("--datasets", nargs="*")
    materialize.add_argument("--no-overwrite", action="store_true")

    baselines = subparsers.add_parser("baselines", help="Show optional baseline availability")
    _ = baselines

    args = parser.parse_args()
    if args.command == "smoke":
        dataset = make_mixed_classification_dataset(
            n_samples=args.samples,
            n_features=args.features,
        )
        config = PipelineConfig(target_column=dataset.target_column)
        if args.screener_device:
            config.screener.device = args.screener_device
        config.screener.progress = args.progress
        config.search.progress = args.progress
        model = NeuralScreenedBNClassifier(config)
        model.fit(dataset.frame, truth_graph=dataset.graph)
        print(json.dumps(model.result_.to_dict(), indent=2))
    elif args.command == "train":
        frame = pd.read_csv(args.csv_path)
        config = PipelineConfig(target_column=args.target)
        if args.screener_device:
            config.screener.device = args.screener_device
        config.screener.progress = args.progress
        config.search.progress = args.progress
        model = NeuralScreenedBNClassifier(config)
        model.fit(frame)
        print(json.dumps(model.result_.to_dict(), indent=2))
    elif args.command == "run-suite":
        config = PipelineConfig(target_column="label")
        if args.screener_device:
            config.screener.device = args.screener_device
        config.screener.progress = not args.no_progress
        config.search.progress = not args.no_progress
        suite_result = run_benchmark_suite(
            suite_name=args.suite_name,
            base_config=config,
            output_path=args.output,
            base_dir=args.base_dir,
            dataset_names=args.datasets,
            baseline_names=args.baselines,
            seeds=args.seeds,
            ablations=args.ablations,
            screener_device=args.screener_device,
            tabpfn_device=args.tabpfn_device,
            progress=not args.no_progress,
        )
        print(json.dumps(suite_result.to_dict(), indent=2))
    elif args.command == "run-evidence":
        config = PipelineConfig(target_column="label")
        if args.screener_device:
            config.screener.device = args.screener_device
        config.screener.progress = not args.no_progress
        config.search.progress = not args.no_progress
        summary = run_staged_evidence_pass(
            base_config=config,
            artifact_dir=args.artifact_dir,
            base_dir=args.base_dir,
            stages=args.stages,
            screener_device=args.screener_device,
            tabpfn_device=args.tabpfn_device,
            progress=not args.no_progress,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "summarize-paper":
        summary = summarize_paper_positioning(args.artifact_dir)
        print(json.dumps(summary, indent=2))
    elif args.command == "materialize-expert-bn":
        summary = materialize_expert_bn_assets(
            root=args.root,
            dataset_names=args.datasets,
            overwrite=not args.no_overwrite,
        )
        print(json.dumps(summary, indent=2))
    else:
        availability = {
            name: {"available": info.available, "detail": info.detail}
            for name, info in external_baseline_availability().items()
        }
        print(json.dumps(availability, indent=2))


if __name__ == "__main__":
    main()
