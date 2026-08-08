"""Loss builder factory module for instantiating loss modules from configuration.

This module provides ``build_loss`` and ``build_loss_function`` to construct
loss modules (CharbonnierLoss or PSNRLoss) from configuration dictionaries or Config objects.
"""

from typing import Any

import torch.nn as nn

from .charbonnier import CharbonnierLoss
from .psnr_loss import PSNRLoss


def build_loss(cfg: Any) -> nn.Module:
    """Construct a loss function module from a configuration object.

    Args:
        cfg: Configuration object or dictionary. Supports nested 'loss' sub-namespace
            or direct parameters.

    Returns:
        nn.Module: Configured CharbonnierLoss or PSNRLoss instance.

    Raises:
        ValueError: If loss type is unsupported or parameters fail validation.
        TypeError: If cfg is invalid type.
    """
    loss_cfg = _extract_loss_config(cfg)

    loss_type = _get_param(loss_cfg, "name", None)
    if loss_type is None:
        loss_type = _get_param(loss_cfg, "type", None)
    if loss_type is None:
        loss_type = _get_param(loss_cfg, "loss_type", "charbonnier")

    if not isinstance(loss_type, str):
        raise ValueError(f"Loss type must be a string, got {type(loss_type).__name__}.")

    loss_type_lower = loss_type.lower().replace("_", "").replace("-", "")

    reduction = _get_param(loss_cfg, "reduction", "mean")

    if loss_type_lower in ("charbonnier", "charbonnierloss"):
        eps = _get_param(loss_cfg, "eps", 1e-3)
        return CharbonnierLoss(eps=float(eps), reduction=reduction)
    elif loss_type_lower in ("psnr", "psnrloss"):
        data_range = _get_param(loss_cfg, "data_range", 1.0)
        eps = _get_param(loss_cfg, "eps", 1e-8)
        return PSNRLoss(
            data_range=float(data_range), eps=float(eps), reduction=reduction
        )
    else:
        raise ValueError(
            f"Unsupported loss type '{loss_type}'. Supported loss types are 'charbonnier' and 'psnr'."
        )


def build_loss_function(cfg: Any) -> nn.Module:
    """Alias for build_loss."""
    return build_loss(cfg)


def _extract_loss_config(cfg: Any) -> Any:
    """Extract loss sub-configuration from a configuration object.

    Args:
        cfg: Configuration dictionary, Config object, or string loss name.

    Returns:
        Loss sub-configuration or dictionary.
    """
    if isinstance(cfg, str):
        return {"name": cfg}
    if isinstance(cfg, dict):
        return cfg.get("loss", cfg)
    if hasattr(cfg, "loss"):
        return cfg.loss
    return cfg


def _get_param(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a parameter from configuration dictionary or object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
