# SEMICON India Hackathon 2026 — KLA Problem Statement 1

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-248%20Passed%20%7C%205%20Skipped-brightgreen?style=for-the-badge)](docs/project-status.md)

> **An end-to-end PyTorch research framework and engineering pipeline employing Nonlinear Activation Free Networks (NAFNet) for joint denoising and $2\times$ super-resolution ($128 \times 128 \to 256 \times 256$) of low-dose Scanning Electron Microscope (SEM) micrographs.**

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Proposed Solution](#proposed-solution)
- [System Restoration Pipeline](#system-restoration-pipeline)
- [Model Architecture](#model-architecture)
- [Layered Software Architecture](#layered-software-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Physics-Inspired Degradation Model](#physics-inspired-degradation-model)
- [Training & Loss Formulations](#training--loss-formulations)
- [Evaluation Metrics & Qualitative Assessment](#evaluation-metrics--qualitative-assessment)
- [Inference & Sliding-Window Pipeline](#inference--sliding-window-pipeline)
- [Verified Technical Results](#verified-technical-results)
- [Historical Experiment Catalog](#historical-experiment-catalog)
- [Reproducibility & Quick Start](#reproducibility--quick-start)
- [Current Submission Status](#current-submission-status)
- [Limitations & Future Work](#limitations--future-work)
- [Documentation Index](#documentation-index)
- [License & Acknowledgements](#license--acknowledgements)

---

## Problem Overview

In modern semiconductor metrology and defect inspection (e.g. sub-7nm node wafer fabrication), **Scanning Electron Microscopy (SEM)** is essential for inspecting nanometer-scale geometries, line-edge roughness (LER), and contact hole profiles.

However, electron beam exposure introduces a fundamental trade-off:
1. **Low Beam Doses**: Preventing photoresist shrinkage, wafer charging, and specimen damage requires minimizing beam current and dwell time.
2. **Compound Degradation**: Low electron counts induce severe **Poisson shot noise**, surface-dependent **multiplicative Gamma speckle noise**, and coarse spatial rastering ($2\times$ downsampling).

Restoration algorithms must recover clean, high-resolution structural information while faithfully preserving nanoscale boundaries without hallucinating false defects.

---

## Proposed Solution

This repository implements an end-to-end, modular restoration framework based on:
- **NAFNet (Nonlinear Activation Free Network)**: Eliminates non-linear activation functions (ReLU, GELU) and heavy self-attention layers, replacing them with linear projections and **SimpleGate** channel multiplication for superior restoration quality and memory efficiency.
- **Joint Denoising & $2\times$ Super-Resolution**: Performs single-pass feature restoration and spatial upscaling via a **PixelShuffle** head and a **global bilinear residual skip connection**.
- **Physics-Calibrated Degradation Modeling**: Directly simulates the physics of low-dose electron emission (Poisson shot noise + Gamma speckle) to align synthetic training data with real-world sensor distributions.
- **Physical Dynamic Range Integrity**: Preserves raw, unclipped float32 sensor values for degraded inputs while enforcing strict $[0.0, 1.0]$ bounds on high-resolution ground-truth targets.

---

## System Restoration Pipeline

```
  NoisyLR SEM (128×128, Raw Unclipped Float32)
                       │
                       ▼
        [Reflection Padding to Multiples of 8]
                       │
                       ▼
     [NAFNet Encoder Stages 0-2 (Width 48)]
                       │
                       ▼
     [NAFNet Bottleneck Stage (Width 384)]
                       │
                       ▼
     [NAFNet Decoder Stages 2'-0' + Lateral Skips]
                       │
                       ▼
      [PixelShuffle 2× Super-Resolution Tail]
                       │
                       ▼
     [+ Global Bilinear Input Residual Addition]
                       │
                       ▼
  Restored High-SNR SEM Micrograph (256×256, [0.0, 1.0])
```

![End-to-End Restoration Pipeline](<assets/diagrams/07d9430c-4a19-43e5-b9f0-e0f9a9ed40d3 - Copy.png>)

*Figure 1: High-level end-to-end restoration pipeline from low-dose ingestion to restored high-resolution output.*

---

## Model Architecture

NAFNet adopts a symmetric U-Net hierarchy tailored for single-channel grayscale SEM restoration:

![NAFNet Architecture](<assets/diagrams/625a251c-3d16-473c-863d-e222335d086c - Copy.png>)

*Figure 2: Complete NAFNet architecture with 3 encoder stages, 1 bottleneck stage, 3 decoder stages, additive skip connections, PixelShuffle $2\times$ SR tail, and global bilinear residual skip.*

### Core Architectural Primitives

![NAFBlock Mechanics](<assets/diagrams/96faf70b-9a3a-4d39-b60f-eb7d15242125 - Copy.png>)

*Figure 3: Internal structure of the Nonlinear Activation Free Block (NAFBlock).*

1. **LayerNorm2d**: Channel-wise normalization adapted for 2D spatial layouts.
2. **SimpleGate**: Replaces non-linear activation functions with element-wise multiplication of split feature channels:
   $$\text{SimpleGate}(X_1, X_2) = X_1 \odot X_2 \quad (X_1, X_2 \in \mathbb{R}^{B \times C \times H \times W})$$
3. **Simplified Channel Attention (SCA)**: Computes global spatial pooling followed by linear channel weighting:
   $$\text{SCA}(X) = X \odot \mathcal{F}_{\text{Linear}}(\text{GAP}(X))$$
4. **Global Bilinear Residual Skip**:
   $$\hat{I}_{\text{restored}} = \text{PixelShuffle}(\text{Tail}(F)) + \text{Bilinear}(I_{\text{input}}, \text{scale}=2)$$

*For detailed architectural specifications, see [docs/architecture.md](docs/architecture.md).*

---

## Layered Software Architecture

The codebase enforces strict modularity and separation of concerns:

![Layered Software Architecture](<assets/diagrams/ea3e9d83-15e4-46c1-9371-f355d3d428ad - Copy.png>)

*Figure 4: Layered software architecture and module interaction boundaries.*

- **Layer 1 (Interface)**: User-facing CLI entrypoints (`train.py`, `scripts/evaluate.py`, `scripts/evaluate_qualitative.py`) and YAML configs.
- **Layer 2 (Core Engine)**: PyTorch execution modules (`src/engine/trainer.py`, `src/engine/evaluator.py`, `src/engine/inference.py`, `src/engine/checkpoint.py`).
- **Layer 3 (Model & Losses)**: NAFNet architectures (`src/models/`) and differentiable losses (`src/losses/`).
- **Layer 4 (Data Pipeline)**: Memory-mapped dataset loaders (`src/datasets/sem_dataset.py`) and Albumentations paired transforms (`src/datasets/transforms.py`).
- **Layer 5 (Storage & Artifacts)**: Native `.npy` storage, checkpoint serialization, and evaluation tables.

---

## Dataset & Preprocessing

![Dataset Preprocessing Pipeline](<assets/diagrams/75fd0087-52d4-48b6-b54b-0d7fc7d3c5ee - Copy.png>)

*Figure 5: Dataset ingestion, intensity handling, spatial augmentation, and memory-mapped collation.*

- **Storage Format**: 32-bit floating-point arrays (`.npy`), avoiding lossy 8-bit dynamic range truncation.
- **Input Dimensions**: NoisyLR $128 \times 128$; Ground Truth $256 \times 256$.
- **Intensity Handling Policy**:
  - **NoisyLR (Unclipped)**: Ingested with `clip=False` to retain true physical sensor noise, negative baseline tails (e.g. down to $-0.002$), and detector spikes ($> 1.0$).
  - **Ground Truth (Clipped)**: Ingested with `clip=True` to constrain targets strictly to $[0.0, 1.0]$.
- **Frozen Split Partitions**: Total 3,200 paired samples partitioned via:
  - Training split (`dataset/train_split.json`): 2,882 samples
  - Validation split (`dataset/val_split.json`): 318 samples
- **Spatial Augmentations**: Synchronized HorizontalFlip ($p=0.5$), VerticalFlip ($p=0.5$), and RandomRotate90 ($p=0.5$).
- **DataLoader Throughput**: Measured sequential throughput of **336.6 samples/sec** with **3.55 ms** P95 fetch latency.

*For complete dataset characterization, see [docs/dataset.md](docs/dataset.md).*

---

## Physics-Inspired Degradation Model

The degradation engine in [`src/datasets/transforms.py`](file:///d:/Programming/python/semicon/src/datasets/transforms.py) simulates physical SEM degradation:

1. **Gaussian Blur**: Optional pre-downsampling blur ($p=0.15$, kernel 3 or 5, $\sigma \in [0.1, 2.0]$).
2. **Bicubic Downsampling ($2\times$)**: OpenCV `cv2.INTER_CUBIC` downsampling without anti-aliasing.
3. **Multiplicative Gamma Speckle**:
   $$I_{\text{speckle}} = I \odot \eta_{\gamma}, \quad \eta_{\gamma} \sim \text{Gamma}(L, 1/L), \quad L = 1/\sigma_s^2, \quad \sigma_s \in [0.040, 0.231]$$
4. **Poisson Shot Noise**:
   $$I_{\text{poisson}} = \frac{\mathcal{P}(\max(I, 0) \cdot S)}{S} + \min(I, 0), \quad S \in [110.0, 180.0]$$
5. **Additive Gaussian Noise**: Configurable thermal noise ($\sigma_g \in [0.0, 0.0183]$, $p=0.0$ in primary `exp009`, available for OOD robustness).

*For complete degradation formulas, see [docs/degradation.md](docs/degradation.md).*

---

## Training & Loss Formulations

![Supervised Training Loop](<assets/diagrams/451a31c0-6670-4eb5-a588-dcf1393f92f0 - Copy.png>)

*Figure 6: Supervised training loop with Charbonnier loss, AdamW optimizer, and Cosine Annealing scheduler.*

- **Optimizer**: AdamW ($\text{lr} = 1.0 \times 10^{-3}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$).
- **Scheduler**: `CosineAnnealingLR` decaying to $\text{min\_lr} = 1.0 \times 10^{-6}$ over 50 epochs.
- **Precision**: PyTorch Automatic Mixed Precision (AMP) with FP16 `GradScaler` and CPU fallback.
- **Numerical Stability**: Finite-loss guard (`torch.isfinite`), finite-gradient checks, and gradient norm clipping (max norm 1.0).
- **Loss Functions**:
  - **Differentiable Charbonnier Loss**: $\mathcal{L}_{\text{Charbonnier}} = \frac{1}{N}\sum \sqrt{(\hat{y}_i - y_i)^2 + \epsilon^2}$ ($\epsilon=10^{-3}$).
  - **SSIM Loss**: Differentiable structural similarity with Gaussian window ($11 \times 11, \sigma=1.5$).
  - **FFT-L1 Loss**: Frequency-domain $L_1$ loss on 2D Fourier magnitudes.
  - **Composite Loss**: $1.0 \times \text{Charbonnier} + 0.2 \times \text{SSIM} + 0.05 \times \text{FFT-L1}$.

> [!NOTE]
> **FFT Normalization Finding**: The current FFT loss implementation uses unnormalized FFT (`norm="backward"`). Because DFT magnitudes scale with image dimensions ($256 \times 256 = 65,536$), the frequency term dominated the composite loss in `exp011`, explaining why composite loss underperformed pure Charbonnier loss (`exp009`).

*For full training details, see [docs/training.md](docs/training.md).*

---

## Evaluation Metrics & Qualitative Assessment

- **PSNR**: Computed with `data_range = 1.0`.
- **SSIM**: 2D Gaussian window ($11 \times 11$, $\sigma=1.5$, `data_range = 1.0`).
- **LPIPS**: AlexNet backbone with 1-to-3 channel replication and $[-1.0, 1.0]$ normalization.
- **Qualitative Comparison**: 4-column residual inspection grids and fine-feature zoom crops.

| Sample ID | 4-Column Qualitative Comparison Grid | Fine Feature Zoom Crop |
| :---: | :---: | :---: |
| **000214** | ![Grid 000214](results/images/qualitative_analysis/000214_comparison_grid.png) | ![Zoom 000214](results/images/qualitative_analysis/000214_zoom_crop.png) |
| **000897** | ![Grid 000897](results/images/qualitative_analysis/000897_comparison_grid.png) | ![Zoom 000897](results/images/qualitative_analysis/000897_zoom_crop.png) |
| **002538** | ![Grid 002538](results/images/qualitative_analysis/002538_comparison_grid.png) | ![Zoom 002538](results/images/qualitative_analysis/002538_zoom_crop.png) |

*For complete metric documentation, see [docs/evaluation.md](docs/evaluation.md).*

---

## Inference & Sliding-Window Pipeline

![Production Inference Engine](<assets/diagrams/a474e8b1-4854-42e0-97f2-16cc47acabbb - Copy.png>)

*Figure 7: Memory-bounded sliding-window inference with 2D Gaussian spatial blending.*

- **Whole-Image Mode**: Reflection padding to multiples of 8, single forward pass, exact $2\times$ unpadding.
- **Sliding-Window Mode**: Tiling with patch size $P=256$, overlap ratio $O=0.25$, and 2D Gaussian spatial accumulation:
  $$W(y, x) = \max\left( \exp\left( -\frac{1}{2}\left[\frac{(y-c_y)^2}{\sigma_y^2} + \frac{(x-c_x)^2}{\sigma_x^2}\right]\right), 10^{-3}\right), \quad \sigma = \frac{P_{\text{out}}}{4}$$
  eliminating grid boundary artifacts across tiles.
- **Architecture Auto-Detection**: Dynamic model configuration inference directly from checkpoint `state_dict`.

*For complete inference details, see [docs/inference.md](docs/inference.md).*

---

## Verified Technical Results

The following table summarizes the **empirically verified** technical metrics across the codebase:

| Category | Specification / Measurement | Value | Status | Verification Source |
| :--- | :--- | :---: | :---: | :--- |
| **Parameters** | NAFNet Width 32 | **1,129,028** (~1.13M) | **VERIFIED** | `scripts/verify_params.py` |
| **Parameters** | NAFNet Width 48 (Primary) | **2,521,444** (~2.52M) | **VERIFIED** | `scripts/verify_params.py` |
| **Parameters** | NAFNet Width 64 | **4,465,796** (~4.47M) | **VERIFIED** | `scripts/verify_params.py` |
| **Complexity** | Width 48 ($128 \times 128 \to 256 \times 256$) | **4.25 GFLOPs** | **VERIFIED** | PyTorch `FlopCounterMode` |
| **Complexity** | Width 48 ($256 \times 256 \to 512 \times 512$) | **17.00 GFLOPs** | **VERIFIED** | PyTorch `FlopCounterMode` |
| **Test Suite** | Unit & Integration Test Suite | **248 Passed / 5 Skipped / 0 Failed** | **VERIFIED** | Pytest (`tests/`) |
| **Data Loader** | Sequential Loader Throughput | **336.6 samples/sec** | **VERIFIED** | `scripts/benchmark_dataset.py` |
| **Data Loader** | P95 Sample Fetch Latency | **3.55 ms** | **VERIFIED** | `scripts/benchmark_dataset.py` |
| **Model Weights** | Trained Final Checkpoint | *Not Included* | **NOT AVAILABLE** | Repository Audit |
| **Validation** | Held-Out Split (`dataset/val_split.json`) | *Not Verified* | **NOT VERIFIED** | Awaiting Checkpoint |
| **Hardware** | Target GPU Latency (T4/H100) | *Not Benchmarked* | **NOT BENCHMARKED** | Awaiting Hardware Run |

---

## Historical Experiment Catalog

> [!IMPORTANT]
> The metrics below represent **historical pre-split experiment logs** recorded during model development. They do **not** represent verified results on the frozen held-out validation split (`dataset/val_split.json`).

| Experiment ID | Architecture Width | Loss Objective | Degradation Pipeline | Historical PSNR | Historical SSIM | Status / Conclusion |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Raw Input** | — | — | — | 22.91 dB | 0.412 | Degraded input reference |
| **Bicubic ($2\times$)** | — | — | — | 25.12 dB | 0.612 | Classical baseline |
| **exp001** | Width 32 | Charbonnier ($L_1$) | Standard | 29.41 dB | 0.789 | Historical Proof-of-concept |
| **exp002** | Width 48 | Charbonnier ($L_1$) | Standard | 29.99 dB | 0.800 | Historical Preferred capacity |
| **exp003** | Width 64 | Charbonnier ($L_1$) | Standard | 30.03 dB | 0.801 | Historical Diminishing returns |
| **exp009** | **Width 48** | **Charbonnier ($L_1$)** | **Physics (Poisson + Gamma)** | **29.15 dB** | **0.864** | **Primary Experimental Target** |
| **exp011** | Width 48 | Composite ($L_1$+SSIM+FFT) | Physics (Poisson + Gamma) | 28.53 dB | 0.860 | Historical Composite finetuning |

*For complete experiment analysis, see [docs/experiments.md](docs/experiments.md).*

---

## Reproducibility & Quick Start

### 1. Installation
```bash
git clone https://github.com/harshwardhan1507/AI-SEM-Image-Restoration.git
cd AI-SEM-Image-Restoration

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Verify Architecture & Parameters
```bash
python scripts/verify_params.py
```

### 3. Run Test Suite
```bash
pytest tests/
```

### 4. Train Primary Model (`exp009`)
```bash
python train.py --config configs/experiments/exp009_augmentation_poisson.yaml
```

### 5. Run Inference
```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/exp009_augmentation_poisson/best_model.pth \
  --input dataset/test/degraded/ \
  --output results/predictions/ \
  --tile-size 256 \
  --overlap 0.25
```

*For complete reproducibility documentation, see [docs/reproducibility.md](docs/reproducibility.md).*

---

## Current Submission Status

### Implemented & Fully Verified
- **NAFNet Architecture**: Width 32/48/64, SimpleGate, SCA, LayerNorm2d, PixelShuffle $2\times$ head, bilinear skip.
- **Dataset Pipeline**: Memory-mapped `.npy` float32 loader with raw unclipped NoisyLR inputs and clipped GT targets.
- **Frozen Split Partitioning**: Defined index splits `dataset/train_split.json` (2,882 samples) and `dataset/val_split.json` (318 samples).
- **Physics Degradation Engine**: Poisson shot noise, multiplicative Gamma speckle, OpenCV bicubic downsampling, configurable Gaussian noise.
- **Training Engine**: PyTorch AdamW, CosineAnnealingLR, AMP mixed precision, gradient clipping, finite-loss guards.
- **Evaluation & Inference**: Full-reference metrics (PSNR, SSIM, LPIPS) and sliding-window Gaussian tile blending.
- **Test Suite**: 248 passed, 5 skipped, 0 failed across 22 test suites.

### Not Included / Not Verified
- **Final Trained Checkpoint**: No pre-trained weights (`.pth` files) are committed in the repository.
- **Frozen Held-Out Validation Metrics**: Metrics on `dataset/val_split.json` have not been computed due to the absent checkpoint.
- **Hardware Benchmarks**: Inference latency (ms), peak VRAM (MB), and throughput on target accelerators (Tesla T4/H100) are not claimed; mathematical FLOP counts are reported instead.

*For the comprehensive technical audit, see [docs/project-status.md](docs/project-status.md).*

---

## Limitations & Future Work

1. **Model Checkpoint Generation**: Train the full 50-epoch baseline model (`exp009`) to generate the official submission checkpoint.
2. **Frozen Held-Out Evaluation**: Run evaluation across `dataset/val_split.json` with the newly trained checkpoint.
3. **Hardware Acceleration Profiling**: Benchmark latency and memory consumption on NVIDIA Tesla T4 and H100 hardware.
4. **Normalized FFT Loss**: Incorporate orthonormal normalization (`norm="ortho"`) in the frequency-domain loss.
5. **Adversarial Loss Integration**: Implement a lightweight PatchGAN discriminator to synthesize high-frequency physical grain textures without spatial over-smoothing.
6. **Edge Deployment Optimization**: ONNX export, TensorRT FP16/INT8 post-training quantization, and Test-Time Augmentation (TTA).

*For a detailed limitations breakdown, see [docs/limitations.md](docs/limitations.md).*

---

## Documentation Index

| Document | Purpose |
| :--- | :--- |
| **[docs/architecture.md](docs/architecture.md)** | Deep technical dive into NAFNet layers, SimpleGate, SCA, parameter counts, and FLOPs. |
| **[docs/dataset.md](docs/dataset.md)** | Paired `.npy` array characterization, unclipped input policy, split files, and throughput. |
| **[docs/degradation.md](docs/degradation.md)** | Physical SEM degradation equations, Poisson-Gamma noise formulation, and exp009 config. |
| **[docs/training.md](docs/training.md)** | Supervised training engine, AdamW, CosineAnnealingLR, AMP, and FFT-L1 scaling analysis. |
| **[docs/evaluation.md](docs/evaluation.md)** | PSNR, SSIM, LPIPS formulations, qualitative residual maps, and verification status. |
| **[docs/inference.md](docs/inference.md)** | Whole-image padding, sliding-window Gaussian blending, and checkpoint auto-detection. |
| **[docs/experiments.md](docs/experiments.md)** | Comprehensive historical experiment catalog (`exp001`-`exp011`) and capacity scaling. |
| **[docs/reproducibility.md](docs/reproducibility.md)** | Complete step-by-step instructions for installation, testing, training, and evaluation. |
| **[docs/limitations.md](docs/limitations.md)** | Transparent analysis of uncommitted checkpoints, unverified metrics, and future roadmap. |
| **[docs/project-status.md](docs/project-status.md)** | Executive technical audit matrix, test suite results, and recent engineering fixes. |

---

## License & Acknowledgements

This repository is distributed under the **[MIT License](LICENSE)**.

- **SEMICON India Hackathon 2026 / KLA Corporation**: For defining Problem Statement 1 and providing technical problem guidance.
- **NAFNet Authors (Megvii Research)**: For pioneering nonlinear activation free neural network architectures (*Chen et al., ECCV 2022*).
- **PyTorch & Albumentations Communities**: For foundational deep learning and computer vision frameworks.