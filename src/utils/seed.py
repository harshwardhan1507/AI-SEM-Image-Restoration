"""Deterministic random seed manager module for SEM NAFNet restoration.

This module provides `set_seed` to enforce deterministic random number generation
across Python built-in random, NumPy, PyTorch CPU, CUDA GPU devices, and cuDNN backends.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Enforce deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value. Defaults to 42.
        deterministic: If True, configures PyTorch cuDNN backend for deterministic
            operation execution.

    Returns:
        int: The active seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed
