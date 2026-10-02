"""Check whether PyTorch can see and use the local GPU."""

from __future__ import annotations

import json

import torch

from complaint_triage.utils.device import get_torch_device_summary


def main() -> None:
    summary = get_torch_device_summary()
    print(json.dumps(summary, indent=2))

    if not torch.cuda.is_available():
        print(
            "\nCUDA is not available to PyTorch. If this machine has an NVIDIA GPU, reinstall "
            "PyTorch using a CUDA-enabled wheel, then run this script again."
        )
        return

    device = torch.device("cuda")
    x = torch.randn((1024, 1024), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(
        f"\nGPU smoke test passed on {torch.cuda.get_device_name(0)}. "
        f"Result mean: {y.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
