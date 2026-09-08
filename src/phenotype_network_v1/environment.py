"""Environment probing for V1 server scaffold and reproducibility.

Detects Python, PyTorch, PyG, CUDA, cuDNN, GPU model, and PyTorch
deterministic settings. PyTorch/PyG are optional at import time so CPU-only
unit tests and the ``cpu`` environment check run on machines without CUDA or
without the ``deep`` dependency group installed.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

try:  # deep dependency group; may be absent on CPU-only machines
    import torch
except Exception:  # pragma: no cover - defensive against broken installs
    torch = None  # type: ignore[assignment]

try:
    import torch_geometric  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    torch_geometric = None  # type: ignore[assignment]


def python_version() -> str:
    """Return the running interpreter version, e.g. ``3.11.9``."""
    return platform.python_version()


def python_implementation() -> str:
    return platform.python_implementation()


def torch_available() -> bool:
    return torch is not None


def pyg_available() -> bool:
    return torch_geometric is not None


def torch_version() -> str | None:
    if torch is None:
        return None
    return torch.__version__


def pyg_version() -> str | None:
    if torch_geometric is None:
        return None
    return torch_geometric.__version__


def cuda_info() -> dict[str, Any]:
    """Report CUDA availability, toolchain, device, and free memory."""
    if torch is None or not torch.cuda.is_available():
        return {
            "available": False,
            "version": None,
            "device_name": None,
            "capability": None,
            "free_memory_bytes": None,
            "total_memory_bytes": None,
        }
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "free_memory_bytes": free,
        "total_memory_bytes": total,
    }


def cudnn_version() -> str | None:
    if torch is None:
        return None
    return torch.backends.cudnn.version()


def deterministic_settings() -> dict[str, Any]:
    """Report PyTorch deterministic configuration flags."""
    if torch is None:
        return {"configured": False}
    return {
        "configured": True,
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


@dataclass
class EnvironmentReport:
    """Structured, JSON-serialisable environment report."""

    ok: bool
    device: str
    python: str = field(default_factory=python_version)
    implementation: str = field(default_factory=python_implementation)
    platform: str = field(default_factory=platform.platform)
    torch_available: bool = field(default_factory=torch_available)
    torch_version: str | None = field(default_factory=torch_version)
    pyg_available: bool = field(default_factory=pyg_available)
    pyg_version: str | None = field(default_factory=pyg_version)
    cuda: dict[str, Any] = field(default_factory=cuda_info)
    cudnn_version: str | None = field(default_factory=cudnn_version)
    deterministic: dict[str, Any] = field(default_factory=deterministic_settings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def check_environment(device: str = "cpu") -> EnvironmentReport:
    """Build the environment report.

    The report is ``ok`` when a ``cpu`` device can be described, or when a
    ``cuda`` device is requested and CUDA plus PyTorch are actually available.
    Missing optional deep-learning components are reported, not fatal, for
    ``cpu`` checks so the scaffold is reproducible before installs finish.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {device!r}; expected 'cpu' or 'cuda'")

    if device == "cuda":
        ok = torch is not None and bool(torch.cuda.is_available())
    else:
        ok = True
    return EnvironmentReport(ok=ok, device=device)


__all__ = [
    "EnvironmentReport",
    "check_environment",
    "cuda_info",
    "cudnn_version",
    "deterministic_settings",
    "pyg_available",
    "pyg_version",
    "python_implementation",
    "python_version",
    "torch_available",
    "torch_version",
]
