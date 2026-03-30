"""Neural dependency screener with bootstrap aggregation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .config import ScreenerConfig
from .data import ColumnInfo, PreparedData
from .models import mutual_information
from .runtime import resolve_torch_device


@dataclass(slots=True)
class InteractionBundle:
    target: str
    sources: Tuple[str, str]
    score: float
    stability: float


@dataclass(slots=True)
class ScreenerResult:
    edge_scores: pd.DataFrame
    edge_stability: pd.DataFrame
    candidate_scores: pd.DataFrame
    candidate_pools: Dict[str, List[str]]
    interaction_bundles: Dict[str, List[InteractionBundle]]


class _SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = hidden_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _MultiTargetNet(nn.Module):
    def __init__(self, input_dim: int, columns: List[ColumnInfo], config: ScreenerConfig) -> None:
        super().__init__()
        self.columns = columns
        self.encoder = _SharedEncoder(input_dim, config.hidden_dim, config.num_layers, config.dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(config.hidden_dim, column.target_size) for column in columns]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project_head(self, encoded: torch.Tensor, target_idx: int) -> torch.Tensor:
        return self.heads[target_idx](encoded)

    def forward_head(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        encoded = self.encode(x)
        return self.project_head(encoded, target_idx)


class NeuralDependencyScreener:
    def __init__(self, config: ScreenerConfig) -> None:
        self.config = config
        self.result_: ScreenerResult | None = None

    def fit(
        self,
        data: PreparedData,
        allowed_sources: Dict[str, List[str]] | None = None,
    ) -> ScreenerResult:
        if self.config.strategy == "mi_filter":
            result = self._fit_mutual_information(data)
            self.result_ = result
            return result
        rng = np.random.default_rng(self.config.random_state)
        device = torch.device(resolve_torch_device(self.config.device))
        x = torch.tensor(data.encoded_matrix, dtype=torch.float32, device=device)
        keep_masks = self._build_keep_masks(data.columns, x.shape[1], device=device, dtype=x.dtype)
        targets = {
            column.name: torch.tensor(data.target_arrays[column.name], device=device)
            for column in data.columns
        }
        total_bootstraps = max(self.config.bootstrap_rounds, 1)
        self._log_progress(
            "starting fit "
            f"device={device} bootstraps={total_bootstraps} epochs={self.config.epochs}"
        )

        edge_runs: List[np.ndarray] = []
        bundle_runs: Dict[tuple[str, tuple[str, str]], List[float]] = {}

        for bootstrap_idx in range(self.config.bootstrap_rounds):
            bootstrap_number = bootstrap_idx + 1
            indices = rng.choice(len(x), size=len(x), replace=True)
            val_size = max(1, int(len(x) * self.config.validation_fraction))
            val_idx = np.unique(indices[:val_size])
            train_idx = indices[val_size:]
            if len(train_idx) == 0:
                train_idx = indices
            self._log_progress(
                f"bootstrap {bootstrap_number}/{total_bootstraps} "
                f"train_rows={len(train_idx)} val_rows={len(val_idx)}"
            )
            model = self._train_single(
                x=x,
                targets=targets,
                columns=data.columns,
                train_idx=train_idx,
                keep_masks=keep_masks,
                seed=self.config.random_state + bootstrap_idx,
                device=device,
                bootstrap_idx=bootstrap_number,
                total_bootstraps=total_bootstraps,
            )
            edge_matrix, bundle_scores = self._score_model(
                model=model,
                x=x[val_idx],
                targets={name: target[val_idx] for name, target in targets.items()},
                columns=data.columns,
                keep_masks=keep_masks,
                allowed_sources=allowed_sources,
            )
            self._log_progress(
                f"bootstrap {bootstrap_number}/{total_bootstraps} scored "
                f"edges={int(np.count_nonzero(edge_matrix > 0))} "
                f"bundles={sum(score > 0 for score in bundle_scores.values())}"
            )
            edge_runs.append(edge_matrix)
            for key, value in bundle_scores.items():
                bundle_runs.setdefault(key, []).append(value)

        edge_stack = np.stack(edge_runs, axis=0)
        edge_scores = np.median(edge_stack, axis=0)
        edge_stability = np.mean(edge_stack > 0, axis=0)

        columns = [column.name for column in data.columns]
        edge_df = pd.DataFrame(edge_scores, index=columns, columns=columns)
        stability_df = pd.DataFrame(edge_stability, index=columns, columns=columns)
        bundles = self._aggregate_bundles(bundle_runs)
        candidate_scores, candidate_pools = self._candidate_pools(columns, edge_df, stability_df, bundles)

        self.result_ = ScreenerResult(
            edge_scores=edge_df,
            edge_stability=stability_df,
            candidate_scores=candidate_scores,
            candidate_pools=candidate_pools,
            interaction_bundles=bundles,
        )
        self._log_progress(
            f"finished fit targets={len(candidate_pools)} "
            f"retained_bundles={sum(len(items) for items in bundles.values())}"
        )
        return self.result_

    def mutual_information_prefilter(
        self,
        data: PreparedData,
        limit: int | None = None,
    ) -> Dict[str, List[str]]:
        edge_df, _ = self._mutual_information_matrices(data)
        source_limit = max(limit or self.config.mi_prefilter_size, 1)
        pools: Dict[str, List[str]] = {}
        for target in edge_df.index:
            ranked = sorted(
                (
                    (source, float(edge_df.loc[target, source]))
                    for source in edge_df.columns
                    if source != target
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            pools[target] = [source for source, _ in ranked[:source_limit]]
        return pools

    def _train_single(
        self,
        x: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        columns: List[ColumnInfo],
        train_idx: np.ndarray,
        keep_masks: torch.Tensor,
        seed: int,
        device: torch.device,
        bootstrap_idx: int,
        total_bootstraps: int,
    ) -> _MultiTargetNet:
        torch.manual_seed(seed)
        model = _MultiTargetNet(x.shape[1], columns, self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        for epoch_idx in range(self.config.epochs):
            shuffled = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
            epoch_loss_total = 0.0
            epoch_batches = 0
            for start in range(0, len(shuffled), self.config.batch_size):
                batch_idx = shuffled[start : start + self.config.batch_size]
                batch_x = x[batch_idx]
                batch_targets = {
                    column.name: targets[column.name][batch_idx]
                    for column in columns
                }
                loss = self._batched_training_loss(
                    model=model,
                    batch_x=batch_x,
                    targets=batch_targets,
                    columns=columns,
                    keep_masks=keep_masks,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_total += float(loss.item())
                epoch_batches += 1
            self._maybe_log_epoch(
                bootstrap_idx=bootstrap_idx,
                total_bootstraps=total_bootstraps,
                epoch_idx=epoch_idx,
                mean_loss=epoch_loss_total / max(epoch_batches, 1),
            )
        model.eval()
        return model

    def _score_model(
        self,
        model: _MultiTargetNet,
        x: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        columns: List[ColumnInfo],
        keep_masks: torch.Tensor,
        allowed_sources: Dict[str, List[str]] | None = None,
    ) -> tuple[np.ndarray, Dict[tuple[str, tuple[str, str]], float]]:
        edge_matrix = np.zeros((len(columns), len(columns)), dtype=float)
        bundle_scores: Dict[tuple[str, tuple[str, str]], float] = {}
        name_to_idx = {column.name: idx for idx, column in enumerate(columns)}

        with torch.no_grad():
            for target_idx, target_info in enumerate(columns):
                allowed_for_target = (
                    set(allowed_sources.get(target_info.name, []))
                    if allowed_sources is not None
                    else None
                )
                source_indices = []
                for source_idx, source_info in enumerate(columns):
                    if source_idx == target_idx:
                        continue
                    if allowed_for_target is not None and source_info.name not in allowed_for_target:
                        continue
                    source_indices.append(source_idx)
                mask_variants = [()] + [(source_idx,) for source_idx in source_indices]
                losses = self._evaluate_target_masks(
                    model=model,
                    x=x,
                    target=targets[target_info.name],
                    columns=columns,
                    target_idx=target_idx,
                    mask_variants=mask_variants,
                    keep_masks=keep_masks,
                )
                baseline = losses[0]
                single_scores: Dict[str, float] = {}
                for source_idx, loss in zip(source_indices, losses[1:]):
                    source_info = columns[source_idx]
                    influence = max(
                        0.0,
                        (loss - baseline) / max(abs(baseline), 1e-6),
                    )
                    edge_matrix[target_idx, source_idx] = influence
                    single_scores[source_info.name] = influence

                if not self.config.use_interaction_bundles or self.config.interaction_top_pairs <= 0:
                    continue
                top_sources = sorted(single_scores, key=single_scores.get, reverse=True)[
                    : self.config.pairwise_top_m
                ]
                pair_variants = [
                    (name_to_idx[left], name_to_idx[right])
                    for left, right in combinations(top_sources, 2)
                ]
                pair_losses = self._evaluate_target_masks(
                    model=model,
                    x=x,
                    target=targets[target_info.name],
                    columns=columns,
                    target_idx=target_idx,
                    mask_variants=pair_variants,
                    keep_masks=keep_masks,
                )
                for (left, right), loss in zip(combinations(top_sources, 2), pair_losses):
                    relative_increase = max(
                        0.0,
                        (loss - baseline) / max(abs(baseline), 1e-6),
                    )
                    synergy = max(0.0, relative_increase - single_scores[left] - single_scores[right])
                    bundle_scores[(target_info.name, tuple(sorted((left, right))))] = synergy

        return edge_matrix, bundle_scores

    def _fit_mutual_information(self, data: PreparedData) -> ScreenerResult:
        columns = [column.name for column in data.columns]
        edge_df, stability_df = self._mutual_information_matrices(data)
        candidate_scores, candidate_pools = self._candidate_pools(columns, edge_df, stability_df, {})
        return ScreenerResult(
            edge_scores=edge_df,
            edge_stability=stability_df,
            candidate_scores=candidate_scores,
            candidate_pools=candidate_pools,
            interaction_bundles={},
        )

    def _mutual_information_matrices(
        self,
        data: PreparedData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = [column.name for column in data.columns]
        edge_scores = np.zeros((len(columns), len(columns)), dtype=float)
        edge_stability = np.zeros((len(columns), len(columns)), dtype=float)
        discrete = data.discrete_frame
        for target_idx, target in enumerate(columns):
            target_values = discrete[target].to_numpy(dtype=int)
            for source_idx, source in enumerate(columns):
                if source == target:
                    continue
                score = max(
                    0.0,
                    mutual_information(
                        discrete[source].to_numpy(dtype=int),
                        target_values,
                    ),
                )
                edge_scores[target_idx, source_idx] = score
                edge_stability[target_idx, source_idx] = 1.0 if score > 0.0 else 0.0
        edge_df = pd.DataFrame(edge_scores, index=columns, columns=columns)
        stability_df = pd.DataFrame(edge_stability, index=columns, columns=columns)
        return edge_df, stability_df

    def _build_keep_masks(
        self,
        columns: List[ColumnInfo],
        input_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        keep_masks = torch.ones((len(columns), input_dim), device=device, dtype=dtype)
        for target_idx, column in enumerate(columns):
            keep_masks[target_idx, column.input_indices] = 0.0
        return keep_masks

    def _batched_training_loss(
        self,
        model: _MultiTargetNet,
        batch_x: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        columns: List[ColumnInfo],
        keep_masks: torch.Tensor,
    ) -> torch.Tensor:
        masked_inputs = batch_x.unsqueeze(0) * keep_masks[:, None, :]
        encoded = model.encode(masked_inputs.reshape(-1, batch_x.shape[1]))
        encoded = encoded.reshape(len(columns), batch_x.shape[0], -1)
        total_loss = torch.tensor(0.0, device=batch_x.device)
        for target_idx, column in enumerate(columns):
            output = model.project_head(encoded[target_idx], target_idx)
            target = targets[column.name]
            if column.kind == "continuous":
                total_loss = total_loss + F.mse_loss(output.squeeze(-1), target.float())
            else:
                total_loss = total_loss + F.cross_entropy(output, target.long())
        return total_loss / max(len(columns), 1)

    def _evaluate_target_masks(
        self,
        model: _MultiTargetNet,
        x: torch.Tensor,
        target: torch.Tensor,
        columns: List[ColumnInfo],
        target_idx: int,
        mask_variants: List[tuple[int, ...]],
        keep_masks: torch.Tensor,
    ) -> List[float]:
        if not mask_variants:
            return []
        losses: List[float] = []
        chunk_size = max(self.config.score_mask_chunk_size, 1)
        base_mask = keep_masks[target_idx]
        target_info = columns[target_idx]
        for start in range(0, len(mask_variants), chunk_size):
            chunk_variants = mask_variants[start : start + chunk_size]
            chunk_masks = []
            for extra_masks in chunk_variants:
                combined_mask = base_mask.clone()
                for source_idx in extra_masks:
                    combined_mask = combined_mask * keep_masks[source_idx]
                chunk_masks.append(combined_mask)
            mask_tensor = torch.stack(chunk_masks, dim=0)
            masked_inputs = x.unsqueeze(0) * mask_tensor[:, None, :]
            encoded = model.encode(masked_inputs.reshape(-1, x.shape[1]))
            output = model.project_head(encoded, target_idx)
            if target_info.kind == "continuous":
                predictions = output.reshape(len(chunk_variants), x.shape[0], -1).squeeze(-1)
                chunk_losses = ((predictions - target.float().unsqueeze(0)) ** 2).mean(dim=1)
            else:
                logits = output.reshape(len(chunk_variants), x.shape[0], -1)
                expanded_target = target.long().unsqueeze(0).expand(len(chunk_variants), -1).reshape(-1)
                per_example = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    expanded_target,
                    reduction="none",
                )
                chunk_losses = per_example.reshape(len(chunk_variants), x.shape[0]).mean(dim=1)
            losses.extend(float(loss.item()) for loss in chunk_losses)
        return losses

    def _masked_loss(
        self,
        model: _MultiTargetNet,
        x: torch.Tensor,
        target: torch.Tensor,
        columns: List[ColumnInfo],
        target_idx: int,
        extra_masks: tuple[int, ...],
        keep_masks: torch.Tensor | None = None,
    ) -> float:
        local_keep_masks = keep_masks
        if local_keep_masks is None:
            local_keep_masks = self._build_keep_masks(
                columns,
                x.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
        return self._evaluate_target_masks(
            model=model,
            x=x,
            target=target,
            columns=columns,
            target_idx=target_idx,
            mask_variants=[extra_masks],
            keep_masks=local_keep_masks,
        )[0]

    def _aggregate_bundles(
        self,
        bundle_runs: Dict[tuple[str, tuple[str, str]], List[float]],
    ) -> Dict[str, List[InteractionBundle]]:
        bundles: Dict[str, List[InteractionBundle]] = {}
        for (target, sources), scores in bundle_runs.items():
            scores_array = np.asarray(scores, dtype=float)
            stability = float(np.mean(scores_array > 0))
            score = float(np.median(scores_array))
            if stability < self.config.min_bundle_stability or score <= 0:
                continue
            bundles.setdefault(target, []).append(
                InteractionBundle(
                    target=target,
                    sources=sources,
                    score=score,
                    stability=stability,
                )
            )
        for target in bundles:
            bundles[target].sort(key=lambda bundle: bundle.score, reverse=True)
            bundles[target] = bundles[target][: self.config.interaction_top_pairs]
        return bundles

    def _maybe_log_epoch(
        self,
        bootstrap_idx: int,
        total_bootstraps: int,
        epoch_idx: int,
        mean_loss: float,
    ) -> None:
        if not self.config.progress:
            return
        interval = max(self.config.progress_interval, 1)
        epoch_number = epoch_idx + 1
        if epoch_number != 1 and epoch_number % interval != 0 and epoch_number != self.config.epochs:
            return
        self._log_progress(
            f"bootstrap {bootstrap_idx}/{total_bootstraps} "
            f"epoch {epoch_number}/{self.config.epochs} loss={mean_loss:.4f}"
        )

    def _log_progress(self, message: str) -> None:
        if not self.config.progress:
            return
        print(f"[screener] {message}", file=sys.stderr, flush=True)

    def _candidate_pools(
        self,
        columns: List[str],
        edge_scores: pd.DataFrame,
        edge_stability: pd.DataFrame,
        bundles: Dict[str, List[InteractionBundle]],
    ) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
        candidate_scores_df = pd.DataFrame(
            np.zeros((len(columns), len(columns)), dtype=float),
            index=columns,
            columns=columns,
        )
        pools: Dict[str, List[str]] = {}
        for target in columns:
            adjusted: Dict[str, float] = {}
            forced_sources: set[str] = set()
            for source in columns:
                if source == target:
                    continue
                stability = float(edge_stability.loc[target, source])
                score = float(edge_scores.loc[target, source])
                if stability >= self.config.min_edge_stability and score > 0:
                    adjusted[source] = score
            for bundle in bundles.get(target, []):
                for source in bundle.sources:
                    adjusted[source] = adjusted.get(source, 0.0) + 0.5 * bundle.score
                    if self.config.force_include_bundle_sources:
                        forced_sources.add(source)
            for source, score in adjusted.items():
                candidate_scores_df.loc[target, source] = float(score)

            ranked = sorted(adjusted.items(), key=lambda item: item[1], reverse=True)
            if not ranked:
                pools[target] = []
                continue

            base_size = max(self.config.candidate_pool_size, 1)
            base_ranked = ranked[:base_size]
            threshold = base_ranked[-1][1]
            selected_sources = {
                source
                for source, score in ranked
                if score >= threshold * self.config.candidate_pool_score_ratio
            }
            selected_sources.update(source for source, _ in base_ranked)
            selected_sources.update(forced_sources)

            capped_ranked = [
                source
                for source, _ in ranked
                if source in selected_sources
            ][: self.config.candidate_pool_soft_cap]
            pools[target] = capped_ranked
        return candidate_scores_df, pools
