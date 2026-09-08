"""Tests for the V1 environment probe (Milestone 0).

These tests must pass on CPU-only machines without PyTorch installed, so the
``deep`` dependency group is never required here.
"""

from __future__ import annotations

import json

import pytest

from phenotype_network_v1 import __version__
from phenotype_network_v1.environment import (
    EnvironmentReport,
    check_environment,
    cuda_info,
    cudnn_version,
    deterministic_settings,
    pyg_available,
    pyg_version,
    python_version,
    torch_available,
    torch_version,
)


def test_package_version_is_defined() -> None:
    assert isinstance(__version__, str) and __version__


def test_python_version_is_reported() -> None:
    version = python_version()
    assert version.count(".") >= 1
    major = int(version.split(".")[0])
    assert major >= 3


def test_cpu_check_always_passes_without_deep_group() -> None:
    report = check_environment("cpu")
    assert report.ok is True
    assert report.device == "cpu"


def test_cuda_check_reports_unavailable_without_gpu() -> None:
    # Never raises: must degrade gracefully when torch/CUDA is absent.
    report = check_environment("cuda")
    if torch_available():
        assert report.ok == bool(report.cuda["available"])
    else:
        assert report.ok is False
    assert report.device == "cuda"


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValueError):
        check_environment("tpu")  # type: ignore[arg-type]


def test_report_contains_required_fields() -> None:
    report = check_environment("cpu")
    payload = report.to_dict()
    for key in (
        "ok",
        "device",
        "python",
        "implementation",
        "platform",
        "torch_available",
        "torch_version",
        "pyg_available",
        "pyg_version",
        "cuda",
        "cudnn_version",
        "deterministic",
    ):
        assert key in payload


def test_report_json_round_trip() -> None:
    report = check_environment("cpu")
    decoded = json.loads(report.to_json())
    assert decoded["ok"] is True
    assert decoded["device"] == "cpu"
    assert decoded["python"] == python_version()


def test_version_functions_agree_with_flags() -> None:
    assert (torch_version() is not None) == torch_available()
    assert (pyg_version() is not None) == pyg_available()


def test_cuda_info_schema() -> None:
    info = cuda_info()
    for key in (
        "available",
        "version",
        "device_name",
        "capability",
        "free_memory_bytes",
        "total_memory_bytes",
    ):
        assert key in info


def test_deterministic_settings_schema() -> None:
    settings = deterministic_settings()
    if torch_available():
        for key in (
            "configured",
            "use_deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
        ):
            assert key in settings
    else:
        assert settings == {"configured": False}


def test_cudnn_version_is_string_when_present() -> None:
    version = cudnn_version()
    assert version is None or isinstance(version, str)


def test_environment_report_dataclass_defaults() -> None:
    report = EnvironmentReport(ok=True, device="cpu")
    assert report.python == python_version()
    assert report.cuda["available"] is False or report.cuda["available"] is True
