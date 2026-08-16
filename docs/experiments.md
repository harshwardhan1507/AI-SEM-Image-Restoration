# Historical Experiment Catalog & Capacity Scaling

## Overview

This document records the empirical experiment history conducted during development. All experiments utilized NAFNet architectures trained with AdamW and Cosine Annealing learning rate schedules.

> [!IMPORTANT]
> **Data Integrity Notice**:
> The metrics below represent **historical pre-split experiment logs** recorded in repository development notes. They do **not** represent verified results on the frozen held-out validation split (`dataset/val_split.json`), as no final trained checkpoint is included in the submitted repository.

---

## Experiment Summary Table

| Experiment ID | Architecture Width | Objective Loss | Degradation Model | Validation PSNR (dB) | Validation SSIM | Status / Conclusion |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Raw Noisy Input** | — | — | — | 22.91 dB | 0.412 | Degraded input reference |
| **Bicubic ($2\times$)** | — | — | — | 25.12 dB | 0.612 | Classical interpolation baseline |
| **exp001** | Width 32 | Charbonnier ($L_1$) | Standard | 29.41 dB | 0.789 | Historical Proof-of-concept |
| **exp002** | Width 48 | Charbonnier ($L_1$) | Standard | 29.99 dB | 0.800 | Historical Preferred capacity baseline |
| **exp003** | Width 64 | Charbonnier ($L_1$) | Standard | 30.03 dB | 0.801 | Historical Diminishing returns study |
| **exp009** | **Width 48** | **Charbonnier ($L_1$)** | **Physics-calibrated (Poisson + Gamma)** | **29.15 dB** | **0.864** | **Primary Experimental Configuration** |
| **exp011** | Width 48 | Composite ($L_1$ + SSIM + FFT) | Physics-calibrated (Poisson + Gamma) | 28.53 dB | 0.860 | Historical Composite loss finetuning |

---

## Detailed Experiment Reports

### 1. Capacity Scaling Study (`exp001`, `exp002`, `exp003`)
- **Objective**: Determine the optimal architectural channel width $C$ balancing restoration fidelity and computational efficiency.
- **Configurations**:
  - `exp001` (Width 32, 1.13M params): 29.41 dB PSNR, 0.789 SSIM.
  - `exp002` (Width 48, 2.52M params): 29.99 dB PSNR (+0.58 dB), 0.800 SSIM (+0.011).
  - `exp003` (Width 64, 4.47M params): 30.03 dB PSNR (+0.04 dB), 0.801 SSIM (+0.001).
- **Finding**: Scaling from Width 32 to 48 yielded substantial quality improvements (+0.58 dB). However, scaling from Width 48 to 64 increased parameter count by +77.1% while providing negligible PSNR gain (+0.04 dB). **Width 48 was established as the Pareto-optimal architecture**.

### 2. Physics-Calibrated Degradation (`exp009`)
- **Config**: [`configs/experiments/exp009_augmentation_poisson.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp009_augmentation_poisson.yaml)
- **Objective**: Replace standard Gaussian assumptions with mathematically calibrated SEM physics: multiplicative Gamma speckle ($\sigma \in [0.040, 0.231]$) and Poisson shot noise ($S \in [110, 180]$).
- **Results**: 29.15 dB PSNR, **0.864 SSIM**.
- **Finding**: Incorporating physically accurate noise models produced a major leap in structural similarity (SSIM increased from 0.800 to 0.864). The model learned to cleanly suppress non-Gaussian sensor speckle while preserving semiconductor pattern edges.

### 3. Multi-Objective Composite Loss Finetuning (`exp011`)
- **Config**: [`configs/experiments/exp011_finetune_ssim.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp011_finetune_ssim.yaml)
- **Objective**: Finetune `exp009` with a composite loss ($1.0 \times \text{Charbonnier} + 0.2 \times \text{SSIM} + 0.05 \times \text{FFT-L1}$) to encourage high-frequency texture reconstruction and reduce $L_1$ smoothing.
- **Results**: 28.53 dB PSNR, 0.860 SSIM.
- **Scientific Analysis**: 
  1. **Unnormalized FFT Term**: The 2D FFT used `norm="backward"`, causing frequency components to scale with $H \times W = 65,536$. This caused the frequency term to overpower the spatial loss terms.
  2. **Capacity Saturation**: At base width 48, the network focuses its representational capacity on low-frequency structures. Reconstructing stochastic high-frequency noise profiles without hallucination requires either higher model capacity or adversarial objectives.

---

## Code References

- Experiment Configurations: [`configs/experiments/`](file:///d:/Programming/python/semicon/configs/experiments/)
- Quantitative Summary Table: [`results/tables/quantitative_results.csv`](file:///d:/Programming/python/semicon/results/tables/quantitative_results.csv)
- Baseline Report: [`experiments/exp001_baseline_report.md`](file:///d:/Programming/python/semicon/experiments/exp001_baseline_report.md)
- Physics Degradation Report: [`experiments/exp009_physics_baseline_report.md`](file:///d:/Programming/python/semicon/experiments/exp009_physics_baseline_report.md)
- Composite Finetuning Report: [`experiments/exp011_composite_finetune_report.md`](file:///d:/Programming/python/semicon/experiments/exp011_composite_finetune_report.md)
