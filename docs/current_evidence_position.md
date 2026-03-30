# Current Evidence Position

This note packages the current benchmark state into a paper-facing summary that is stable in the repo, rather than only living in generated artifacts.

The underlying benchmark artifacts are:

- [artifacts/benchmarks/paper_minimum_full.json](/home/yugam/classifier/artifacts/benchmarks/paper_minimum_full.json)
- [artifacts/benchmarks/evidence_summary.json](/home/yugam/classifier/artifacts/benchmarks/evidence_summary.json)
- [artifacts/benchmarks/paper_positioning.md](/home/yugam/classifier/artifacts/benchmarks/paper_positioning.md)
- [artifacts/benchmarks/paper_results.md](/home/yugam/classifier/artifacts/benchmarks/paper_results.md)
- [artifacts/benchmarks/known_graph_fast_probe.json](/home/yugam/classifier/artifacts/benchmarks/known_graph_fast_probe.json)

## Bottom Line

The current default method should be framed as a **quality-first explicit BN classifier**.

- It passes the staged structural rule: `3/5` known-graph wins against the best explicit BN baseline on SHD.
- It passes the staged predictive-competitiveness rule: `2/2` real datasets are within the configured ROC-AUC gap of strong non-BN baselines.
- It fails the staged runtime rule: `0/5` runtime wins against `greedy_hc_bn`.

So the honest paper position is:

- strong structural result among explicit BN methods,
- competitive real-data predictive behavior,
- runtime still substantially weaker than `greedy_hc_bn`.

## What The Evidence Supports

### Structural story

The method is already credible on the structural side.

- Strong wins on `insurance`
- Win on `child`
- Win on `synthetic_small`
- Near-tie on `alarm`
- Loss on `synthetic_medium`

This is enough to support a quality-first explicit-BN claim, even though it is not enough to support a speed-first claim.

### Predictive story

The method remains close to strong predictive baselines on the real datasets that matter in the staged evidence.

- `sklearn_breast_cancer`: within the configured competitiveness gap of the best strong baseline
- `sklearn_wine`: within the configured competitiveness gap of the best strong baseline

That means the paper can honestly say the method preserves predictive competitiveness while keeping an explicit DAG.

### Runtime story

Runtime is the blocker.

- The current default loses to `greedy_hc_bn` on all `5/5` known-graph datasets
- The runtime ratios are large, not marginal
- This is the only staged criterion preventing a `go` verdict

The repo should therefore avoid claiming the current method is a practical speed replacement for `greedy_hc_bn`.

## What The Fast Probe Says

The `known_graph_fast` ablation and the separate fast-mode redesign probe are useful evidence, but they do not justify changing the default.

- `known_graph_fast` narrows the runtime gap a lot, but still loses to `greedy_hc_bn` on `5/5` known-graph datasets
- It also introduces SHD regressions on multiple datasets, so it is not promotable
- The broader `fast_mode_design_probe` found no promotion winner among:
  - `known_graph_fast`
  - `mi_filter_fast`
  - `single_pass_neural_fast`
  - `hybrid_mi_neural_fast`

The important conclusion is that **incremental runtime tuning has now been exhausted**. Any future runtime effort should be treated as a separate redesign track, not as a tweak to the current default.

## Recommended Paper Framing

Use language like this:

> Neural-screened BN is a quality-first explicit BN classifier that improves structural performance over standard explicit BN baselines on key known-graph benchmarks while remaining predictively competitive with strong tabular baselines on real datasets.

Use limitation language like this:

> The current method does not yet satisfy a strict runtime criterion against lightweight hill-climbing BN search. Faster ablations reduce cost substantially, but none preserve enough structure quality while clearing the runtime gate.

Avoid language like this:

- “faster than classical BN search”
- “practical replacement for greedy hill-climb BN”
- “best overall tabular classifier”

## Novelty Positioning

### Abstract / Introduction Wording

> We present a quality-first explicit Bayesian network classifier that combines neural dependency screening with candidate-constrained BN structure search. The method uses a learned screener to construct stable candidate parent pools, then performs explicit DAG selection under structural and predictive objectives, yielding an interpretable classifier rather than a black-box predictor. Empirically, it improves structural quality over standard explicit BN baselines on key known-graph benchmarks while remaining predictively competitive on real tabular datasets. The contribution is therefore a new end-to-end explicit-BN classification pipeline, not a claim to invent neural structure learning or candidate-restricted BN search from scratch.

### Related Work / Reviewer Wording

> The novelty of this work is best understood as system-level synthesis rather than a brand-new learning paradigm. Prior work has studied candidate-restricted and hybrid BN structure learning, discriminative BN classifier structure learning, and neural DAG discovery separately. Our contribution is to integrate these ideas into a quality-first explicit BN classifier pipeline in which a neural screener proposes stable parent candidates and a constrained explicit BN search selects the final interpretable DAG. To our knowledge, this specific combination, evaluated as an explicit-BN classification system with both structural and predictive evidence, is not the main focus of prior work.

Safe claim: We introduce a quality-first explicit BN classifier that couples neural dependency screening with constrained BN search to improve structural quality while preserving predictive competitiveness.

Claims to avoid:

- `first neural BN structure learner`
- `first candidate-pruned BN learner`
- `first discriminative BN classifier learner`

## Recommended Project Direction

The default method should stay frozen for now.

- Keep `default` as the quality-first configuration
- Keep `known_graph_fast` and the other fast modes as benchmark-only ablations
- Use the current result set for paper tables, README positioning, and discussion
- Only reopen method changes if there is appetite for a **separate radical runtime redesign**

If the project continues on the paper track rather than the redesign track, the next work should focus on:

1. cleaning and curating paper tables/figures,
2. tightening claims and limitations around the current evidence,
3. preserving the frozen benchmark artifacts as the project baseline.
