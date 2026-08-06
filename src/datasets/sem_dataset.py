"""SEM image dataset loader module for PyTorch pipelines.

This module provides `SEMDataset`, a PyTorch `torch.utils.data.Dataset` subclass
implementing lazy memory-mapped array loading, pixel intensity clipping, shape
formatting, and sample dictionary output.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .scanner import DatasetPair, DatasetScanner
from .validator import DatasetValidator


class SEMDataset(Dataset):
    """PyTorch Dataset for paired SEM restoration micrographs."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        clip_range: Tuple[float, float] = (0.0, 1.0),
        transform: Optional[Callable] = None,
        validate: bool = True,
    ) -> None:
        """Initialize SEMDataset.

        Args:
            root_dir: Path to root dataset directory.
            split: Dataset split ('train' or 'test').
            clip_range: Pixel intensity clipping range (min, max).
            transform: Optional augmentation transform callable.
            validate: If True, executes header validation on scanned pairs.
        """
        super().__init__()
        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.clip_range = clip_range
        self.transform = transform

        self.scanner = DatasetScanner(self.root_dir)
        self.pairs = self.scanner.scan_split(split)

        if validate:
            validator = DatasetValidator()
            validator.validate_dataset_index(self.pairs)

    def __len__(self) -> int:
        """Return total number of dataset samples.

        Returns:
            int: Number of items in dataset split.
        """
        return len(self.pairs)

    def _process_array(self, file_path: Path) -> torch.Tensor:
        """Load, clip, and format a single .npy file into a PyTorch tensor.

        Args:
            file_path: Path to `.npy` array file.

        Returns:
            torch.Tensor: Formatted tensor of shape `(1, H, W)` and dtype `float32`.
        """
        arr_mmap = np.load(file_path, mmap_mode="r")
        arr = np.array(arr_mmap, dtype=np.float32)

        if self.clip_range is not None:
            arr = np.clip(arr, self.clip_range[0], self.clip_range[1])

        tensor = torch.from_numpy(arr)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[torch.Tensor, str, Optional[torch.Tensor]]]:
        """Retrieve a single dataset sample by integer index.

        Args:
            index: Dataset sample index.

        Returns:
            Dict containing:
                - "input": NoisyLR input tensor (1, 128, 128)
                - "target": Ground Truth target tensor (1, 256, 256) or None
                - "filename": Sample ID string
        """
        pair: DatasetPair = self.pairs[index]
        input_tensor = self._process_array(pair.input_path)

        target_tensor: Optional[torch.Tensor] = None
        if pair.target_path is not None:
            target_tensor = self._process_array(pair.target_path)

        if self.transform is not None:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": pair.sample_id,
        }
