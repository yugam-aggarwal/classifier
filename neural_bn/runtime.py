"""Runtime helpers for device selection and temporary environment overrides."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator


def resolve_torch_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def benchmark_runtime_env(
    *,
    artifact_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, str]:
    current = os.environ.get("MPLCONFIGDIR")
    if current:
        mplconfig_dir = Path(current).expanduser()
    elif artifact_dir is not None:
        mplconfig_dir = Path(artifact_dir) / ".runtime" / "mplconfig"
    elif output_path is not None:
        mplconfig_dir = Path(output_path).parent / ".runtime" / "mplconfig"
    else:
        mplconfig_dir = Path("/tmp/neural_bn_mplconfig")
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    return {"MPLCONFIGDIR": str(mplconfig_dir)}
