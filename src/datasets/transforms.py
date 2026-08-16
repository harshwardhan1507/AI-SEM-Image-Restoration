"""Paired spatial data augmentation and synthetic degradation module for SEM image restoration.

This module provides synchronized spatial transformation pipelines using `albumentations` 
for paired low-resolution noisy input micrographs (128x128) and high-resolution ground 
truth targets (256x256). It also includes a physics-based synthetic degradation pipeline
to generate out-of-distribution (OOD) training samples from Ground Truth images.
"""

from typing import Any, Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch


class SyntheticDegradation:
    """Physics-based synthetic degradation pipeline for SEM images."""

    def __init__(
        self,
        blur_prob: float = 0.15,
        speckle_prob: float = 1.0,
        poisson_prob: float = 1.0,
        gaussian_prob: float = 0.0,
        speckle_sigma_range: Tuple[float, float] = (0.0455, 0.2405),
        poisson_scale_range: Tuple[float, float] = (100.0, 250.0),
        gaussian_sigma_range: Tuple[float, float] = (0.0, 0.0183),
    ) -> None:
        """Initialize the synthetic degradation parameters.

        Args:
            blur_prob: Probability of applying Gaussian blur before downsampling.
            speckle_prob: Probability of applying Gamma speckle noise.
            poisson_prob: Probability of applying Poisson (shot) noise.
            gaussian_prob: Probability of applying additive Gaussian noise (for OOD robustness per PS explainer).
            speckle_sigma_range: Range for uniform sampling of Gamma noise standard deviation.
            poisson_scale_range: Range for Poisson scaling factor (1/b).
            gaussian_sigma_range: Range for uniform sampling of additive Gaussian noise standard deviation.
        """
        self.blur_prob = blur_prob
        self.speckle_prob = speckle_prob
        self.poisson_prob = poisson_prob
        self.gaussian_prob = gaussian_prob
        self.speckle_sigma_range = speckle_sigma_range
        self.poisson_scale_range = poisson_scale_range
        self.gaussian_sigma_range = gaussian_sigma_range

    def __call__(self, hr_image: np.ndarray, target_image: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply synthetic degradation to a high-resolution ground truth image.

        Pipeline order:
        1. Renormalize input to [0, 1]
        2. Optional Gaussian Blur
        3. Bicubic downsample (no anti-aliasing)
        4. Optional Gamma multiplicative speckle noise
        5. Optional Poisson shot noise (measured var ∝ μ)
        6. Optional Additive Gaussian noise (OOD insurance)
        
        Args:
            hr_image: High resolution image array (H, W).
            target_image: High resolution target image array (H, W). If provided, it will be renormalized to [0, 1] as well.
            
        Returns:
            If target_image is provided, returns (synthetic_lr, renormalized_target).
            Otherwise, returns synthetic_lr.
        """
        # 1. Renormalize input to [0, 1]
        hr_min, hr_max = hr_image.min(), hr_image.max()
        if hr_max > hr_min:
            hr_image = (hr_image - hr_min) / (hr_max - hr_min)
        else:
            hr_image = np.zeros_like(hr_image)
            
        if target_image is not None:
            t_min, t_max = target_image.min(), target_image.max()
            if t_max > t_min:
                target_image = (target_image - t_min) / (t_max - t_min)
            else:
                target_image = np.zeros_like(target_image)

        # 2. Optional Gaussian Blur (insurance against alternative OOD downsampling)
        if np.random.rand() < self.blur_prob:
            # Random kernel size 3 or 5, random sigma
            ksize = int(np.random.choice([3, 5]))
            sigma = np.random.uniform(0.1, 2.0)
            hr_image = cv2.GaussianBlur(hr_image, (ksize, ksize), sigmaX=sigma)

        # 3. Downsample x2 with INTER_CUBIC (no anti-aliasing)
        h, w = hr_image.shape
        lr_shape = (w // 2, h // 2)
        lr_image = cv2.resize(hr_image, lr_shape, interpolation=cv2.INTER_CUBIC)

        # 4. Optional Gamma multiplicative speckle noise
        if np.random.rand() < self.speckle_prob:
            sigma_s = np.random.uniform(*self.speckle_sigma_range)
            L = 1.0 / (sigma_s ** 2)
            noise_gamma = np.random.gamma(shape=L, scale=1.0/L, size=lr_image.shape).astype(np.float32)
            lr_image = lr_image * noise_gamma
            
        # 5. Optional Poisson shot noise
        if np.random.rand() < self.poisson_prob:
            scale = np.random.uniform(*self.poisson_scale_range)
            # Preserve the negative tail by separating positive and negative parts
            clean_pos = np.clip(lr_image, 0, None)
            clean_neg = np.clip(lr_image, None, 0)
            noisy_pos = np.random.poisson(clean_pos * scale) / scale
            lr_image = (noisy_pos + clean_neg).astype(np.float32)
            
        # 6. Optional Additive Gaussian noise (OOD insurance for PS explainer "dissimilar test data")
        if np.random.rand() < self.gaussian_prob:
            sigma_g = np.random.uniform(*self.gaussian_sigma_range)
            noise_gauss = np.random.normal(0, sigma_g, size=lr_image.shape).astype(np.float32)
            lr_image = lr_image + noise_gauss

        if target_image is not None:
            return lr_image, target_image
        return lr_image


class PairedTransforms:
    """Synchronized spatial data augmentations and synthetic degradation for SEM."""

    def __init__(
        self,
        is_train: bool = True,
        synth_prob: float = 0.0,
        synth_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize PairedTransforms pipeline.

        Args:
            is_train: If True, builds random spatial augmentation pipeline.
            synth_prob: Probability of replacing the real LR input with a synthesized one.
            synth_kwargs: Configuration for the SyntheticDegradation pipeline.
        """
        self.is_train = is_train
        self.synth_prob = synth_prob
        
        if self.is_train:
            self.spatial_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={"target": "image"},
                is_check_shapes=False,
            )
        else:
            self.spatial_transform = None

        if synth_kwargs is None:
            synth_kwargs = {}
        self.synthetic_degradation = SyntheticDegradation(**synth_kwargs)

    def __call__(
        self,
        input_image: Union[np.ndarray, torch.Tensor],
        target_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sample_index: int = 0,
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]
    ]:
        """Apply transformations and optional synthetic degradation.

        Args:
            input_image: Input array or tensor of shape (H, W) or (1, H, W).
            target_image: Target array or tensor of shape (H, W) or (1, H, W), if available.
            sample_index: Index of the sample, used for deterministic RNG in validation synthesis.

        Returns:
            Tuple containing transformed (input_image, target_image).
        """
        is_input_tensor = isinstance(input_image, torch.Tensor)
        is_target_tensor = isinstance(target_image, torch.Tensor) if target_image is not None else False

        input_np = input_image.squeeze(0).numpy() if is_input_tensor else input_image
        target_np = target_image.squeeze(0).numpy() if is_target_tensor else target_image

        # 1. Spatial Transforms (Geometric Augmentation)
        if self.spatial_transform is not None:
            if target_np is not None:
                augmented = self.spatial_transform(image=input_np, target=target_np)
                trans_input_np = augmented["image"]
                trans_target_np = augmented["target"]
            else:
                augmented = self.spatial_transform(image=input_np)
                trans_input_np = augmented["image"]
                trans_target_np = None
        else:
            trans_input_np = input_np
            trans_target_np = target_np

        # 2. Synthetic Degradation
        if target_np is not None and self.synth_prob > 0.0:
            # Use deterministic RNG for validation/testing if not training
            if not self.is_train:
                # Save current numpy state
                state = np.random.get_state()
                # Seed deterministically based on sample_index
                np.random.seed(sample_index)
                
                if np.random.rand() < self.synth_prob:
                    trans_input_np, trans_target_np = self.synthetic_degradation(trans_target_np, trans_target_np)
                
                # Restore original numpy state
                np.random.set_state(state)
            else:
                if np.random.rand() < self.synth_prob:
                    trans_input_np, trans_target_np = self.synthetic_degradation(trans_target_np, trans_target_np)

        # Convert back to tensor if required
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


def get_transforms(
    is_train: bool = True,
    synth_prob: float = 0.0,
    synth_kwargs: Optional[Dict[str, Any]] = None,
) -> PairedTransforms:
    """Construct PairedTransforms instance.

    Args:
        is_train: If True, enables spatial augmentations.
        synth_prob: Probability of synthesizing LR input.
        synth_kwargs: Configuration for SyntheticDegradation.

    Returns:
        PairedTransforms: Configured transformation pipeline.
    """
    return PairedTransforms(is_train=is_train, synth_prob=synth_prob, synth_kwargs=synth_kwargs)
