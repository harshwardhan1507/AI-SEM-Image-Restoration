"""Paired spatial data augmentation module for SEM image restoration.

This module provides synchronized spatial transformation pipelines (flips, 90-degree
orthogonal rotations) using `albumentations` for paired low-resolution noisy input
micrographs (128x128) and high-resolution ground truth targets (256x256).
"""

from typing import Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch


class PairedTransforms:
    """Synchronized spatial data augmentations for paired SEM micrographs."""

    def __init__(self, is_train: bool = True) -> None:
        """Initialize PairedTransforms pipeline.

        Args:
            is_train: If True, builds random spatial augmentation pipeline;
                otherwise returns identity pass-through.
        """
        self.is_train = is_train
        if self.is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={"target": "image"},
                is_check_shapes=False,
            )
        else:
            self.transform = None

    def __call__(
        self,
        input_image: Union[np.ndarray, torch.Tensor],
        target_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]
    ]:
        """Apply synchronized spatial transformations to input and target images.

        Args:
            input_image: Input array or tensor of shape (H, W) or (1, H, W).
            target_image: Target array or tensor of shape (H, W) or (1, H, W), if available.

        Returns:
            Tuple containing transformed (input_image, target_image).
        """
        if not self.is_train or self.transform is None:
            return input_image, target_image

        is_input_tensor = isinstance(input_image, torch.Tensor)
        is_target_tensor = (
            isinstance(target_image, torch.Tensor)
            if target_image is not None
            else False
        )

        input_np = input_image.squeeze(0).numpy() if is_input_tensor else input_image
        target_np = (
            target_image.squeeze(0).numpy()
            if is_target_tensor and target_image is not None
            else target_image
        )

        if target_np is not None:
            augmented = self.transform(image=input_np, target=target_np)
            trans_input_np = augmented["image"]
            trans_target_np = augmented["target"]
        else:
            augmented = self.transform(image=input_np)
            trans_input_np = augmented["image"]
            trans_target_np = None

        if is_input_tensor:
            trans_input = torch.from_numpy(np.ascontiguousarray(trans_input_np))
            if trans_input.ndim == 2:
                trans_input = trans_input.unsqueeze(0)
        else:
            trans_input = trans_input_np

        if target_image is not None:
            if is_target_tensor:
                trans_target = torch.from_numpy(np.ascontiguousarray(trans_target_np))
                if trans_target.ndim == 2:
                    trans_target = trans_target.unsqueeze(0)
            else:
                trans_target = trans_target_np
        else:
            trans_target = None

        return trans_input, trans_target


def get_transforms(is_train: bool = True) -> PairedTransforms:
    """Construct PairedTransforms instance.

    Args:
        is_train: If True, enables spatial augmentations.

    Returns:
        PairedTransforms: Configured transformation pipeline.
    """
    return PairedTransforms(is_train=is_train)
