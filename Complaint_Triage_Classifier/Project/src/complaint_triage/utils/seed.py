"""Random seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducible experiments.

    With ``deterministic=True`` cuDNN autotuning is disabled so repeated GPU
    runs match each other. That costs some throughput; set it to False if you
    care more about speed than bitwise reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        # Torch is not needed for the preprocessing and labeling steps.
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
