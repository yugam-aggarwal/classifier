# Neural-Screened Explicit BN Classifiers

Research scaffold for classification-first Bayesian network structure learning with:

- a lightweight neural dependency screener,
- bootstrap-stable candidate parent pools,
- higher-order interaction bundles,
- candidate-constrained BN structure search, and
- explicit discrete BN baselines such as `NB`, `TAN`, `KDB`, and `AODE`.

This repo is intentionally a research scaffold, not a finished paper package yet. The current benchmark path now includes canonical expert-network structural assets plus strong tabular baselines, while heavier model families such as `TabM` and `TabICL` remain extension points.

## What Is Implemented

- `neural_bn.screener`: shared multi-target MLP screener with bootstrap aggregation.
- `neural_bn.search`: greedy candidate-constrained BN structure search with mixed generative and discriminative scoring.
- `neural_bn.models`: discrete BN inference plus `Naive Bayes`, `TAN`, `KDB`, and `AODE`.
- `neural_bn.pipeline`: one end-to-end estimator that fits the screener, searches the graph, fits the final BN, and reports predictive plus structural metrics.
- `neural_bn.synthetic`: synthetic mixed-type dataset generator for smoke tests.
- `neural_bn.datasets`: dataset adapters for synthetic known-graph data, sklearn real datasets, CSV datasets, and optional local expert-BN datasets.
- `neural_bn.experiments`: benchmark runners, ablations, suite execution, and machine-readable result tables.
- `neural_bn.results`: serializable experiment records with metadata, runtimes, graphs, candidate pools, and split details.

## Quick Start

Run the synthetic smoke test:

```bash
python3 -m neural_bn smoke --samples 400 --features 6 --progress
```

Train on a local CSV:

```bash
python3 -m neural_bn train data.csv --target label
```

See which optional external baselines are currently available:

```bash
python3 -m neural_bn baselines
```

Run a benchmark suite and emit machine-readable results:

```bash
python3 -m neural_bn run-suite smoke --output smoke_results.json
```

Progress logs stream to stderr during `run-suite` and `run-evidence`, so stdout and saved JSON artifacts stay machine-readable. Pass `--no-progress` if you want a quiet run.

Run the paper-oriented suite:

```bash
python3 -m neural_bn run-suite paper_minimum --output paper_minimum_results.json
```

Run the staged evidence workflow and write stable artifacts under `artifacts/benchmarks/`:

```bash
python3 -m neural_bn run-evidence
```

Generate the paper-facing summary from the frozen benchmark artifacts without rerunning benchmarks:

```bash
python3 -m neural_bn summarize-paper --artifact-dir artifacts/benchmarks
```

Run only the lighter stages first:

```bash
python3 -m neural_bn run-evidence --stages sanity_default structural_default
```

If you have GPUs, you can pin the screener and `TabPFN` separately:

```bash
python3 -m neural_bn run-evidence \
  --stages sanity_default structural_default \
  --screener-device cuda:0 \
  --tabpfn-device cuda:1
```

If you want a quiet artifact-only run, add `--no-progress`.

Regenerate the canonical expert-network assets locally:

```bash
python3 -m neural_bn materialize-expert-bn
```

Run a smaller custom suite slice:

```bash
python3 -m neural_bn run-suite smoke \
  --datasets synthetic_tiny sklearn_iris \
  --baselines neural_screened_bn naive_bayes \
  --seeds 5 \
  --ablations default
```

## Benchmark Support

- Named suites: `smoke`, `paper_minimum`
- Built-in structural datasets: `synthetic_tiny`, `synthetic_small`, `synthetic_medium`
- Built-in real datasets: `sklearn_breast_cancer`, `sklearn_iris`, `sklearn_wine`
- Bundled local expert-BN datasets: `alarm`, `child`, `insurance`

Local expert-BN datasets are expected under `datasets/expert_bn/<name>/` with:

- `frame.csv`: tabular samples including the target column
- `edges.csv`: edge list with columns `source,target`
- `metadata.json` optional: `target_column`, `source`, and `notes`

Unavailable optional datasets are skipped gracefully and recorded in the suite output.

This repo now includes deterministic expert-network assets for `alarm`, `child`, and `insurance` generated from `pgmpy.get_example_model(...)` with canonical structures and CPDs. The bundled `frame.csv` files are reproducible samples from those expert networks rather than hand-written placeholders, and can be regenerated with `python3 -m neural_bn materialize-expert-bn`.

## Result Schema

Every experiment record now includes:

- dataset name, regime, split id, seed, and ablation label
- predictive metrics and structural metrics
- runtime breakdown and search statistics
- serialized graph edges and candidate parent pools
- split metadata for train, validation, and test sizes
- explicit skipped-item reporting for unavailable datasets or baseline failures

The staged evidence workflow writes:

- `artifacts/benchmarks/sanity_default.json`
- `artifacts/benchmarks/structural_default.json`
- `artifacts/benchmarks/paper_minimum_full.json`
- `artifacts/benchmarks/evidence_summary.json`

The summary file includes per-stage tables plus a fixed go/no-go decision based on structural wins, runtime wins against `greedy_hc_bn`, predictive competitiveness within `0.03` ROC-AUC of strong baselines, and benchmark health checks for `TabPFN` plus expert-network coverage.

## Current Evidence Snapshot

The current default should be read as a quality-first research mode, not a speed-first replacement for `greedy_hc_bn`.

- `default`: the main quality-first configuration. On the current `paper_minimum_full` evidence run it passes the staged structural and predictive-competitiveness rules, but still fails the runtime rule against `greedy_hc_bn`.
- `known_graph_fast`: an experimental runtime ablation for known-graph datasets. It narrows the runtime gap substantially, but it is not the recommended default because it still misses the runtime gate and introduces dataset-specific SHD regressions.

The repo now includes `artifacts/benchmarks/paper_positioning.json` and `artifacts/benchmarks/paper_positioning.md` as the paper-facing summary layer on top of the canonical benchmark artifacts. The same `summarize-paper` command also writes publication-oriented outputs at `artifacts/benchmarks/paper_results.json` and `artifacts/benchmarks/paper_results.md`.

For a checked-in, repo-stable interpretation of the current result, see [docs/current_evidence_position.md](/home/yugam/classifier/docs/current_evidence_position.md).

## Notes

- The final search currently operates on a discrete table. Continuous features are standardized for the screener and discretized with supervised tree-based bins for BN search.
- The screener now defaults to `auto` device selection and will use CUDA when PyTorch can see a GPU. You can override this with `--screener-device` or by setting `PipelineConfig.screener.device`.
- The long-running benchmark commands now emit progress to stderr at the stage, dataset, baseline, bootstrap, and periodic epoch levels. `smoke` and `train` only show screener progress when you pass `--progress`.
- The prototype prioritizes explicit graphs, inspectable candidate pools, and a clean extension path for future benchmarks.
- Structural metrics such as `SHD` are only meaningful when you pass a ground-truth graph.
- External baselines such as `XGBoost`, `LightGBM`, and `CatBoost` run when installed; otherwise they are reported as unavailable or skipped.
- `TabPFN` is wired into the suite and prefers `NEURAL_BN_TABPFN_MODEL_PATH` when set. Otherwise it auto-discovers checkpoints from `NEURAL_BN_TABPFN_CACHE_DIR`, `~/.cache/tabpfn`, and `/tmp/neural_bn_tabpfn` before falling back to the package default behavior. You can pin its runtime device with `--tabpfn-device` or `NEURAL_BN_TABPFN_DEVICE`.
