"""Utilities for reporting PyTorch CPU/GPU availability."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TorchDeviceSummary:
    """Small, serializable summary of the PyTorch execution environment."""

    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    selected_device: str
    device_name: str | None


def get_torch_device_summary(force_cpu: bool = False) -> dict[str, object]:
    """Return a concise summary of the available PyTorch device.

    Hugging Face ``Trainer`` automatically places the model on CUDA when a CUDA-enabled
    PyTorch build is installed and CPU execution is not forced. This helper makes that
    behavior visible in logs and in the standalone GPU check script.
    """

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    if force_cpu or not cuda_available:
        selected_device = "cpu"
        device_name = None
    else:
        current_device = torch.cuda.current_device()
        selected_device = f"cuda:{current_device}"
        device_name = torch.cuda.get_device_name(current_device)

    return asdict(
        TorchDeviceSummary(
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda,
            device_count=device_count,
            selected_device=selected_device,
            device_name=device_name,
        )
    )
