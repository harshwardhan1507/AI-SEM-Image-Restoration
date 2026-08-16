# Known Limitations & Future Work

## Overview

In alignment with technical rigor and honesty, this document explicitly outlines the current state, known engineering limitations, and planned future extensions for the SEM image restoration repository.

---

## Current Technical Limitations

### 1. Absence of Committed Model Checkpoints
- **Status**: **NOT INCLUDED IN SUBMITTED REPOSITORY**.
- **Explanation**: While the NAFNet architecture, training engine, loss functions, data loader, and inference modules are fully implemented and covered by 248 passing unit tests, no pre-trained weights (`.pth`, `.pt`, `.ckpt`, `.onnx`) are included in `weights/` or `outputs/checkpoints/`.
- **Impact**: Users and evaluators must run `train.py` to produce a trained checkpoint before running inference with `scripts/evaluate.py`.

### 2. Unverified Held-Out Split Metrics
- **Status**: **NOT VERIFIED**.
- **Explanation**: The dataset includes frozen split index files (`dataset/train_split.json` with 2,882 samples and `dataset/val_split.json` with 318 samples). However, because no trained checkpoint is stored, final held-out validation metrics (PSNR, SSIM, LPIPS) on `val_split.json` are not verified.
- **Distinction**: All numerical metrics reported in `docs/experiments.md` (`exp001` through `exp011`) are historical pre-split experiment logs and must not be interpreted as frozen held-out validation results.

### 3. Hardware Latency & Throughput Benchmarks
- **Status**: **NOT BENCHMARKED ON TARGET GPU HARDWARE**.
- **Explanation**: Empirical inference latency (ms/frame), peak VRAM consumption, and hardware throughput on production accelerators (e.g., NVIDIA Tesla T4 or H100) are not available in this submission.
- **Verified Alternative**: Model computational complexity is documented strictly via mathematical FLOP counts measured using PyTorch `FlopCounterMode` (4.25 GFLOPs at $128 \times 128 \to 256 \times 256$).

### 4. Composite FFT Loss Scaling
- **Status**: **IDENTIFIED SCIENTIFIC BOTTLENECK**.
- **Explanation**: The frequency-domain loss in `src/losses/composite.py` employs unnormalized 2D FFT (`norm="backward"`). As DFT magnitudes scale with $H \times W = 256 \times 256 = 65,536$, frequency components dominated the gradient updates in `exp011`, explaining why composite loss underperformed pure Charbonnier loss (`exp009`).

---

## Future Research & Engineering Roadmap

### 1. Loss Formulation Improvements
- **Normalized FFT Loss**: Transition to `norm="ortho"` in `torch.fft.fft2` or scale the frequency weight by $1/(HW)$ to ensure balanced gradient magnitudes across spatial and spectral domains.
- **Adversarial (GAN) Objectives**: Implement a lightweight PatchGAN discriminator to penalize structural over-smoothing and restore stochastic high-frequency semiconductor grain without hallucinating false defects.

### 2. Capacity & Architectural Extensions
- **Wider Bottleneck Variants**: Explore width configurations beyond 64 with selective channel gating to increase capacity for complex nanoscale topographies.
- **Test-Time Augmentation (TTA)**: Implement 8-fold geometric ensemble averaging ($0^\circ, 90^\circ, 180^\circ, 270^\circ$ rotation $\times$ horizontal flips) during inference to boost PSNR by an estimated $+0.2$ to $+0.4$ dB.

### 3. Fab Edge Deployment & Acceleration
- **ONNX & TensorRT Compilation**: Export trained NAFNet models to ONNX graph format and compile with NVIDIA TensorRT FP16/INT8 engines for sub-millisecond inline inspection.
- **Post-Training Quantization (PTQ)**: Quantize weights and activations to INT8 with calibration on representative SEM arrays.
