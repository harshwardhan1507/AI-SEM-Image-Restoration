"""Dataset integrity validator module for SEM image restoration.

This module provides validation logic and custom exception classes to verify the
integrity, array shapes, data types, and pairing consistency of SEM dataset samples.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from src.datasets.scanner import DatasetPair


class DatasetValidationError(Exception):
    """Base exception class for dataset validation failures."""

    pass


class InvalidShapeError(DatasetValidationError):
    """Exception raised when an array has an unexpected spatial shape."""

    pass


class InvalidDtypeError(DatasetValidationError):
    """Exception raised when an array has an unexpected data type."""

    pass


class DatasetValidator:
    """Validator class for checking SEM dataset array shapes, dtypes, and pairings."""

    def __init__(
        self,
        gt_shape: Tuple[int, ...] = (256, 256),
        noisy_shape: Tuple[int, ...] = (128, 128),
        expected_dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize DatasetValidator with expected array dimensions and data types.

        Args:
            gt_shape: Expected 2D spatial dimensions for Ground Truth arrays.
            noisy_shape: Expected 2D spatial dimensions for NoisyLR arrays.
            expected_dtype: Expected NumPy numerical data type.
        """
        self.gt_shape = gt_shape
        self.noisy_shape = noisy_shape
        self.expected_dtype = np.dtype(expected_dtype)

    def validate_array_header(
        self, file_path: Path, expected_shape: Tuple[int, ...]
    ) -> None:
        """Validate NumPy array shape and data type using memory mapping header read.

        Args:
            file_path: Path to `.npy` array file.
            expected_shape: Expected 2D or 3D shape tuple.

        Raises:
            FileNotFoundError: If array file does not exist on disk.
            InvalidDtypeError: If array dtype does not match expected_dtype.
            InvalidShapeError: If array shape does not match expected_shape.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Array file does not exist: {file_path}")

        try:
            arr = np.load(file_path, mmap_mode="r")
        except Exception as err:
            raise DatasetValidationError(
                f"Failed to load array header from {file_path}: {err}"
            ) from err

        if arr.dtype != self.expected_dtype:
            raise InvalidDtypeError(
                f"Invalid data type for {file_path.name}: "
                f"expected {self.expected_dtype}, got {arr.dtype}."
            )

        # Handle both (H, W) and (1, H, W) channel dimensions
        shape_2d = arr.shape[-2:]
        if shape_2d != expected_shape:
            raise InvalidShapeError(
                f"Invalid spatial dimensions for {file_path.name}: "
                f"expected spatial shape {expected_shape}, got full shape {arr.shape}."
            )

    def validate_pair(self, pair: DatasetPair) -> bool:
        """Validate an individual DatasetPair item.

        Args:
            pair: DatasetPair instance to validate.

        Returns:
            bool: True if validation succeeds.

        Raises:
            DatasetValidationError: If validation fails.
        """
        self.validate_array_header(pair.input_path, self.noisy_shape)

        if pair.target_path is not None:
            self.validate_array_header(pair.target_path, self.gt_shape)

        return True

    def validate_dataset_index(self, pairs: Sequence[DatasetPair]) -> bool:
        """Validate a list of DatasetPair samples and check for duplicate IDs.

        Args:
            pairs: Sequence of DatasetPair instances.

        Returns:
            bool: True if all pairs in dataset index pass validation.

        Raises:
            DatasetValidationError: If duplicate sample IDs or file errors are detected.
        """
        seen_ids = set()
        for pair in pairs:
            if pair.sample_id in seen_ids:
                raise DatasetValidationError(
                    f"Duplicate sample ID detected in dataset index: '{pair.sample_id}'"
                )
            seen_ids.add(pair.sample_id)
            self.validate_pair(pair)

        return True
