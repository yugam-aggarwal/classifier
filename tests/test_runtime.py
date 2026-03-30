from __future__ import annotations

from neural_bn.runtime import resolve_torch_device


def test_resolve_torch_device_honors_explicit_request():
    assert resolve_torch_device("cuda:1") == "cuda:1"
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_auto_returns_supported_string():
    assert resolve_torch_device("auto") in {"cpu", "mps", "cuda:0"}
