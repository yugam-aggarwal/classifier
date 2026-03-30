"""Materialize canonical expert-BN benchmark assets from pgmpy example models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class ExpertBNSpec:
    name: str
    target_column: str
    sample_size: int
    sample_seed: int


def expert_bn_specs() -> Dict[str, ExpertBNSpec]:
    return {
        "alarm": ExpertBNSpec(name="alarm", target_column="PRESS", sample_size=2500, sample_seed=1701),
        "child": ExpertBNSpec(
            name="child",
            target_column="LowerBodyO2",
            sample_size=2500,
            sample_seed=1703,
        ),
        "insurance": ExpertBNSpec(
            name="insurance",
            target_column="CarValue",
            sample_size=2500,
            sample_seed=1705,
        ),
    }


def materialize_expert_bn_assets(
    root: Path,
    dataset_names: Sequence[str] | None = None,
    overwrite: bool = True,
) -> Dict[str, Dict[str, str | int]]:
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.utils import get_example_model

    specs = expert_bn_specs()
    requested = list(dataset_names or specs.keys())
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise KeyError(f"Unknown expert BN datasets: {', '.join(unknown)}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    original_pgmpy_level = logging.getLogger("pgmpy").level
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    written: Dict[str, Dict[str, str | int]] = {}
    try:
        for name in requested:
            spec = specs[name]
            dataset_root = root / name
            frame_path = dataset_root / "frame.csv"
            edges_path = dataset_root / "edges.csv"
            metadata_path = dataset_root / "metadata.json"
            if not overwrite and frame_path.exists() and edges_path.exists() and metadata_path.exists():
                written[name] = {
                    "status": "skipped_existing",
                    "target_column": spec.target_column,
                    "sample_size": spec.sample_size,
                    "sample_seed": spec.sample_seed,
                    "root": str(dataset_root),
                }
                continue

            model = get_example_model(name)
            sampler = BayesianModelSampling(model)
            frame = sampler.forward_sample(
                size=spec.sample_size,
                seed=spec.sample_seed,
                show_progress=False,
            )
            ordered_columns = [column for column in frame.columns if column != spec.target_column] + [
                spec.target_column
            ]
            frame = frame.loc[:, ordered_columns]
            edges = pd.DataFrame(model.edges(), columns=["source", "target"])
            metadata = {
                "target_column": spec.target_column,
                "source": f"pgmpy::example_model::{name}",
                "notes": [
                    f"Deterministic sample from pgmpy.get_example_model('{name}') with canonical structure and CPDs.",
                    f"Sample size: {spec.sample_size}. Sampling seed: {spec.sample_seed}.",
                    f"Classification target column: {spec.target_column}.",
                ],
            }

            dataset_root.mkdir(parents=True, exist_ok=True)
            frame.to_csv(frame_path, index=False)
            edges.to_csv(edges_path, index=False)
            metadata_path.write_text(json.dumps(metadata, indent=2))
            written[name] = {
                "status": "written",
                "target_column": spec.target_column,
                "sample_size": spec.sample_size,
                "sample_seed": spec.sample_seed,
                "root": str(dataset_root),
            }
    finally:
        logging.getLogger("pgmpy").setLevel(original_pgmpy_level)

    return written
