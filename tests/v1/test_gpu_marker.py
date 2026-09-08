"""GPU-marked integration test (Milestone 8).

Collects under ``pytest -m gpu`` and is skipped when CUDA is absent, so the
server acceptance command ``pytest -q -m gpu`` runs without CUDA too.
"""

from __future__ import annotations

import pytest

import torch


@pytest.mark.gpu
def test_cuda_available_on_gpu_server() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)
