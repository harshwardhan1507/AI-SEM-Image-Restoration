"""DataLoader builder and deterministic worker initialization module.

This module provides high-performance, reproducible, configuration-driven PyTorch
DataLoader construction for SEM restoration datasets.
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import Config

from .collate import sem_collate
from .sem_dataset import SEMDataset


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader.

    Enforces deterministic random seeding across Python built-in `random`,
    `numpy.random`, and `torch` inside each DataLoader worker process.

    Args:
        worker_id: Worker process rank integer assigned by DataLoader.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _extract_config_value(
    config_data: Dict[str, Any], keys: List[str], default: Any
) -> Any:
    """Extract a configuration value trying multiple candidate key paths.

    Args:
        config_data: Flattened or nested configuration dictionary.
        keys: List of dot-delimited key path strings to try.
        default: Default value if no key path is found.

    Returns:
        Any: First resolved key value or default.
    """
    for key in keys:
        parts = key.split(".")
        curr: Any = config_data
        found = True
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                found = False
                break
        if found and curr is not None:
            return curr
    return default


def validate_dataloader_params(
    dataset: Any,
    batch_size: int,
    num_workers: int,
) -> None:
    """Validate dataset instance and DataLoader numeric parameters.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size per iteration.
        num_workers: Number of worker subprocesses.

    Raises:
        ValueError: If dataset is None, dataset has zero length, batch_size <= 0,
            or num_workers < 0.
    """
    if dataset is None:
        raise ValueError("Dataset instance cannot be None.")

    if hasattr(dataset, "__len__"):
        try:
            ds_len = len(dataset)
        except TypeError:
            ds_len = None

        if ds_len == 0:
            raise ValueError("Dataset is empty (len=0). Cannot construct DataLoader.")

    if batch_size <= 0:
        raise ValueError(
            f"Invalid batch_size ({batch_size}). Must be an integer greater than 0."
        )
    if num_workers < 0:
        raise ValueError(
            f"Invalid num_workers ({num_workers}). Must be a non-negative integer."
        )


def _resolve_dataloader_options(  # noqa: C901
    cfg_dict: Dict[str, Any],
    split: str,
    batch_size: Optional[int],
    num_workers: Optional[int],
    shuffle: Optional[bool],
    drop_last: Optional[bool],
    pin_memory: Optional[bool],
    persistent_workers: Optional[bool],
    prefetch_factor: Optional[int],
    seed: Optional[int],
    timeout: Optional[float],
) -> Dict[str, Any]:
    """Resolve and extract individual DataLoader option parameters."""
    res_batch_size = (
        batch_size
        if batch_size is not None
        else _extract_config_value(
            cfg_dict,
            [
                f"data.{split}_batch_size",
                f"{split}_batch_size",
                "data.batch_size",
                "train.batch_size",
                "batch_size",
            ],
            default=4,
        )
    )

    res_num_workers = (
        num_workers
        if num_workers is not None
        else _extract_config_value(
            cfg_dict,
            [
                "data.num_workers",
                "dataloader.num_workers",
                "train.num_workers",
                "num_workers",
            ],
            default=4,
        )
    )

    res_shuffle = (
        shuffle
        if shuffle is not None
        else _extract_config_value(
            cfg_dict,
            [f"data.{split}_shuffle", f"{split}_shuffle", "shuffle"],
            default=(split == "train"),
        )
    )

    res_drop_last = (
        drop_last
        if drop_last is not None
        else _extract_config_value(
            cfg_dict,
            [f"data.{split}_drop_last", f"{split}_drop_last", "drop_last"],
            default=(split == "train"),
        )
    )

    if pin_memory is None:
        pin_memory_cfg = _extract_config_value(
            cfg_dict,
            ["data.pin_memory", "dataloader.pin_memory", "pin_memory"],
            default=True,
        )
        res_pin_memory = (
            torch.cuda.is_available()
            if isinstance(pin_memory_cfg, str) and pin_memory_cfg.lower() == "auto"
            else bool(pin_memory_cfg)
        )
    else:
        res_pin_memory = pin_memory

    res_persistent_workers = (
        persistent_workers
        if persistent_workers is not None
        else _extract_config_value(
            cfg_dict,
            [
                "data.persistent_workers",
                "dataloader.persistent_workers",
                "persistent_workers",
            ],
            default=True,
        )
    )

    res_prefetch_factor = (
        prefetch_factor
        if prefetch_factor is not None
        else _extract_config_value(
            cfg_dict,
            ["data.prefetch_factor", "dataloader.prefetch_factor", "prefetch_factor"],
            default=2,
        )
    )

    res_seed = (
        seed
        if seed is not None
        else _extract_config_value(
            cfg_dict,
            ["system.seed", "seed"],
            default=42,
        )
    )

    res_timeout = (
        timeout
        if timeout is not None
        else _extract_config_value(
            cfg_dict,
            ["data.timeout", "dataloader.timeout", "timeout"],
            default=0.0,
        )
    )

    return {
        "batch_size": res_batch_size,
        "num_workers": res_num_workers,
        "shuffle": res_shuffle,
        "drop_last": res_drop_last,
        "pin_memory": res_pin_memory,
        "persistent_workers": res_persistent_workers,
        "prefetch_factor": res_prefetch_factor,
        "seed": res_seed,
        "timeout": float(res_timeout),
    }


def build_dataloader(
    dataset: Dataset,
    config: Optional[Union[Config, Dict[str, Any]]] = None,
    split: str = "train",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    seed: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    timeout: Optional[float] = None,
) -> DataLoader:
    """Build a single PyTorch DataLoader with deterministic seeding and performance options."""
    cfg_dict = config.to_dict() if isinstance(config, Config) else (config or {})

    opts = _resolve_dataloader_options(
        cfg_dict=cfg_dict,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        timeout=timeout,
    )

    # Validate dataset and parameters
    validate_dataloader_params(
        dataset=dataset,
        batch_size=opts["batch_size"],
        num_workers=opts["num_workers"],
    )

    # PyTorch constraints: persistent_workers and prefetch_factor require num_workers > 0
    if opts["num_workers"] == 0:
        opts["persistent_workers"] = False
        opts["prefetch_factor"] = None

    if collate_fn is None:
        collate_fn = sem_collate

    # Construct PyTorch generator for deterministic sampling
    generator = torch.Generator()
    generator.manual_seed(opts["seed"])

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": opts["batch_size"],
        "shuffle": opts["shuffle"],
        "drop_last": opts["drop_last"],
        "num_workers": opts["num_workers"],
        "pin_memory": opts["pin_memory"],
        "collate_fn": collate_fn,
        "generator": generator,
        "worker_init_fn": seed_worker if opts["num_workers"] > 0 else None,
        "timeout": opts["timeout"],
    }

    if opts["num_workers"] > 0:
        kwargs["persistent_workers"] = opts["persistent_workers"]
        if opts["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = opts["prefetch_factor"]

    return DataLoader(**kwargs)


def _detect_splits(
    root_path: Path, configured_splits: Optional[List[str]]
) -> List[str]:
    """Detect available dataset splits or parse configured split list."""
    if configured_splits:
        return list(configured_splits)

    candidate_splits = ["train", "val", "test"]
    detected: List[str] = []
    for s in candidate_splits:
        s_dir = root_path / s
        if s_dir.exists():
            detected.append(s)
        elif s == "train" and (root_path / "train" / "train").exists():
            detected.append(s)
        elif s == "test" and (root_path / "Test_NoisyLR").exists():
            detected.append(s)

    return detected if detected else ["train", "test"]


def build_dataloaders(
    config: Union[Config, Dict[str, Any]],
    dataset_dir: Optional[Union[str, Path]] = None,
    splits: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Build all requested split DataLoaders driven by configuration."""
    cfg_dict = config.to_dict() if isinstance(config, Config) else config

    if dataset_dir is None:
        dataset_dir = _extract_config_value(
            cfg_dict,
            ["data.dataset_dir", "dataset_dir", "data.dataset_path", "dataset_path"],
            default="./datasets",
        )

    root_path = Path(dataset_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

    configured_splits = _extract_config_value(cfg_dict, ["data.splits", "splits"], None)
    target_splits = (
        splits if splits is not None else _detect_splits(root_path, configured_splits)
    )

    loaders: Dict[str, DataLoader] = {}

    for split in target_splits:
        dataset_split = split
        if split == "val" and not (root_path / "val").exists():
            dataset_split = "train"

        try:
            dataset = SEMDataset(root_dir=root_path, split=dataset_split)
            loaders[split] = build_dataloader(
                dataset=dataset,
                config=config,
                split=split,
            )
        except (FileNotFoundError, ValueError) as err:
            if splits is not None and split in splits:
                raise ValueError(
                    f"Failed to construct DataLoader for split '{split}': {err}"
                ) from err

    if not loaders:
        raise ValueError(
            f"No valid DataLoaders could be constructed from root_dir: {root_path}"
        )

    return loaders
