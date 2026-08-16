# Experiment 011: Composite Loss Finetuning (SSIM + FFT)

**Date**: August 2026  
**Status**: COMPLETED  
**Architecture**: NAFNet (Width 48)  
**Objective**: Finetune the `exp009` baseline using a perceptual and frequency-aware loss function to combat L1 over-smoothing and restore high-frequency nanoscale textures.

## Hypothesis
Since pure Charbonnier (L1) loss regresses stochastic grain to the mean, it results in structurally sound but perceptually smooth/blurry images. We hypothesized that freezing the broad structural understanding (by loading the `exp009` checkpoint) and finetuning with a **Composite Loss** (L1 + SSIM + FFT) would force the network to reconstruct sharp structural edges and high-frequency textures without hallucinating structures.

## Methodology
1. **Checkpoint**: Loaded `outputs/checkpoints/exp009_augmentation_poisson/best_model.pth`.
2. **Loss Function**: `CompositeLoss` combining:
   - **Charbonnier (Weight 1.0)**: Maintains structural integrity.
   - **SSIM (Weight 0.2)**: Penalizes structural mismatch to sharpen edges.
   - **FFT-L1 (Weight 0.05)**: Penalizes frequency spectrum differences to restore high-frequency grain.
3. **Training**: Finetuned for 50 epochs with a reduced initial learning rate using AdamW + Cosine Annealing.

## Results
- **Validation PSNR**: 28.53 dB (Slightly lower than the 29.15 dB baseline, which is expected as perceptual losses inherently trade-off pixel-perfect MSE/PSNR for perceptual sharpness).
- **Validation SSIM**: 0.860
- **Visual Quality**: 
  - Empirical evaluation over the held-out test sets showed **no noticeable visual divergence** in sharpness from the L1 baseline.
  - The expected restoration of stochastic physical grain did not materialize.

## Honest Scientific Conclusion & Analysis
Despite the theoretical backing of Composite Loss, the empirical results suggest that the NAFNet bottleneck width of 48 heavily biases the network toward low-frequency structural representations. 
L1 loss appears to be already extracting the absolute maximum structural capacity from this architecture constraint. When asked to represent high-frequency noise profiles via SSIM/FFT, the network simply lacks the internal parameter dimensionality (capacity) to model those complex high-frequency textures while simultaneously maintaining the primary structure.

**Future Work**:
To achieve true perceptual sharpness and grain restoration on this dataset, future iterations must either:
1. Drastically increase the NAFNet width (e.g., beyond 64).
2. Adopt a GAN-based adversarial loss (e.g., using a discriminator network) which is far more aggressive at forcing the generator to hallucinate perceptually convincing textures.
