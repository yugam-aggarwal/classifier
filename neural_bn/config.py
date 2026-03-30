"""Configuration objects for the neural BN pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class ScreenerConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 25
    bootstrap_rounds: int = 3
    validation_fraction: float = 0.2
    candidate_pool_size: int = 7
    candidate_pool_soft_cap: int = 10
    candidate_pool_score_ratio: float = 0.8
    force_include_bundle_sources: bool = True
    pairwise_top_m: int = 6
    interaction_top_pairs: int = 5
    score_mask_chunk_size: int = 16
    min_edge_stability: float = 0.5
    min_bundle_stability: float = 0.34
    device: str = "auto"
    progress: bool = False
    progress_interval: int = 5
    random_state: int = 7
    use_interaction_bundles: bool = True
    strategy: Literal["neural", "mi_filter", "hybrid_mi_neural"] = "neural"
    mi_prefilter_size: int = 5


@dataclass(slots=True)
class SearchConfig:
    max_parents: int = 3
    max_additions_per_child: int = 3
    max_iters: int = 50
    generative_score: str = "bic"
    equivalent_sample_size: float = 10.0
    generative_weight: float = 1.0
    discriminative_weight: float = 1.0
    laplace: float = 1.0
    warm_start: Literal["multi_start", "screener_sparse", "naive_bayes", "empty"] = "multi_start"
    multi_start_candidates: tuple[Literal["naive_bayes", "screener_sparse", "empty"], ...] = (
        "naive_bayes",
        "screener_sparse",
    )
    normalize_objective_by_samples: bool = True
    edge_prior_weight: float = 0.5
    hub_penalty_weight: float = 0.1
    improve_tol: float = 1e-6
    validation_checkpoint_interval: int = 5
    validation_patience: int = 3
    min_delta_per_sample: float = 0.01
    validation_prune_passes: int = 2
    validation_prune_passes_real: int = 1
    validation_prune_log_loss_tol: float = 0.002
    validation_prune_roc_auc_tol: float = 0.001
    validation_prune_log_loss_tol_real: float = 0.001
    validation_prune_roc_auc_tol_real: float = 0.0005
    validation_prune_max_fraction_known_graph: float = 0.30
    validation_prune_max_fraction_real: float = 0.35
    validation_prune_min_edges_known_graph: int = 20
    validation_prune_min_edges_real: int = 8
    validation_prune_support_protect: float = 0.75
    validation_prune_target_support_protect: float = 0.60
    progress: bool = False
    proposal_progress_interval: int = 100


@dataclass(slots=True)
class PipelineConfig:
    target_column: str
    n_bins: int = 5
    min_samples_per_bin: int = 20
    random_state: int = 7
    use_screener: bool = True
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass(slots=True)
class AblationConfig:
    label: str = "default"
    disable_screener: bool = False
    disable_interaction_bundles: bool = False
    disable_bootstrap: bool = False
    known_graph_fast: bool = False
    fast_mode_strategy: Literal["mi_filter_fast", "single_pass_neural_fast", "hybrid_mi_neural_fast"] | None = None
    objective: str = "mixed"
    candidate_pool_size: int | None = None
    max_parents: int | None = None
    warm_start: Literal["multi_start", "screener_sparse", "naive_bayes", "empty"] | None = None
    edge_prior_weight: float | None = None
    validation_prune_passes: int | None = None
    validation_prune_passes_real: int | None = None
    validation_prune_support_protect: float | None = None

    def apply(
        self,
        config: PipelineConfig,
        dataset_regime: str = "unknown",
        num_columns: int | None = None,
    ) -> PipelineConfig:
        updated = copy.deepcopy(config)
        updated.random_state = updated.screener.random_state = self._effective_seed(updated)

        if self.disable_screener:
            updated.use_screener = False
        if self.disable_interaction_bundles:
            updated.screener.use_interaction_bundles = False
            updated.screener.interaction_top_pairs = 0
        if self.disable_bootstrap:
            updated.screener.bootstrap_rounds = 1
            updated.screener.min_edge_stability = 0.0
            updated.screener.min_bundle_stability = 0.0
        if self.objective == "generative":
            updated.search.generative_weight = 1.0
            updated.search.discriminative_weight = 0.0
        elif self.objective == "discriminative":
            updated.search.generative_weight = 0.0
            updated.search.discriminative_weight = 1.0
        if self.candidate_pool_size is not None:
            updated.screener.candidate_pool_size = self.candidate_pool_size
        if self.max_parents is not None:
            updated.search.max_parents = self.max_parents
        if self.warm_start is not None:
            updated.search.warm_start = self.warm_start
        if self.edge_prior_weight is not None:
            updated.search.edge_prior_weight = self.edge_prior_weight
        if self.validation_prune_passes is not None:
            updated.search.validation_prune_passes = self.validation_prune_passes
        if self.validation_prune_passes_real is not None:
            updated.search.validation_prune_passes_real = self.validation_prune_passes_real
        if self.validation_prune_support_protect is not None:
            updated.search.validation_prune_support_protect = self.validation_prune_support_protect
        if self.known_graph_fast and dataset_regime == "known_graph":
            updated.screener.bootstrap_rounds = 1
            updated.screener.epochs = 15
            updated.screener.candidate_pool_size = 3
            updated.screener.candidate_pool_soft_cap = 3
            updated.screener.use_interaction_bundles = True
            updated.search.warm_start = "naive_bayes"
            updated.search.multi_start_candidates = ("naive_bayes",)
            updated.search.max_iters = 12 if (num_columns or 0) <= 20 else 20
            updated.search.validation_prune_passes = 0
        if self.fast_mode_strategy and dataset_regime == "known_graph":
            updated.screener.epochs = 15
            updated.screener.candidate_pool_size = 3
            updated.screener.candidate_pool_soft_cap = 3
            if self.fast_mode_strategy == "mi_filter_fast":
                updated.screener.strategy = "mi_filter"
            elif self.fast_mode_strategy == "single_pass_neural_fast":
                updated.screener.strategy = "neural"
                updated.screener.bootstrap_rounds = 1
                updated.screener.min_edge_stability = 0.0
                updated.screener.min_bundle_stability = 0.0
                updated.screener.use_interaction_bundles = False
                updated.screener.interaction_top_pairs = 0
                updated.search.warm_start = "naive_bayes"
                updated.search.multi_start_candidates = ("naive_bayes",)
                updated.search.max_iters = 12 if (num_columns or 0) <= 20 else 20
            elif self.fast_mode_strategy == "hybrid_mi_neural_fast":
                updated.screener.strategy = "hybrid_mi_neural"
                updated.screener.bootstrap_rounds = 1
                updated.screener.min_edge_stability = 0.0
                updated.screener.min_bundle_stability = 0.0
                updated.screener.mi_prefilter_size = 5
        return updated

    @staticmethod
    def _effective_seed(config: PipelineConfig) -> int:
        return config.random_state
