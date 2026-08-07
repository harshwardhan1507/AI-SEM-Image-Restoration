"""Custom collate function module for PyTorch DataLoader pipelines.

This module provides `sem_collate`, a specialized collate function for batching
SEM dataset samples (dictionaries containing input tensors, optional target tensors,
and metadata) without breaking when target tensors are None in test/inference splits.
"""

from typing import Any, Dict, List

import torch


def sem_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of sample dictionaries into batched tensors and metadata lists.

    Args:
        batch: List of sample dictionaries returned by `SEMDataset.__getitem__`.
            Each dictionary contains:
                - "input": torch.Tensor of shape (C, H, W)
                - "target": torch.Tensor of shape (C, H, W) or None
                - "filename": str sample identifier
                - optional additional metadata keys.

    Returns:
        Dict[str, Any]: Batched sample dictionary containing:
            - "input": torch.Tensor of shape (B, C, H, W)
            - "target": torch.Tensor of shape (B, C, H, W) or None
            - "filename": List[str] of sample identifiers
            - additional batched metadata keys.

    Raises:
        ValueError: If batch is empty.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    elem_keys = batch[0].keys()
    collated: Dict[str, Any] = {}

    for key in elem_keys:
        values = [sample[key] for sample in batch]

        # Check if all values for this key are None (e.g., target in test split)
        if all(v is None for v in values):
            collated[key] = None
        # If values are PyTorch Tensors, stack them into a single batch tensor
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        # If values are strings, preserve as a list of strings
        elif isinstance(values[0], str):
            collated[key] = values
        # If values are numbers (int/float/bool) or other primitives, convert to tensor
        elif isinstance(values[0], (int, float, bool)):
            collated[key] = torch.tensor(values)
        # For nested dictionaries or complex metadata, recurse or collect list
        elif isinstance(values[0], dict):
            collated[key] = sem_collate(values)
        else:
            collated[key] = values

    return collated
