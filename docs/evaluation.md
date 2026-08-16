# Evaluation Metrics & Qualitative Assessment

## Overview

The evaluation suite is implemented in [`src/metrics/`](file:///d:/Programming/python/semicon/src/metrics/) and [`src/engine/evaluator.py`](file:///d:/Programming/python/semicon/src/engine/evaluator.py). It provides full-reference quantitative metrics (PSNR, SSIM, LPIPS) and automated qualitative visualization tools ([`scripts/evaluate_qualitative.py`](file:///d:/Programming/python/semicon/scripts/evaluate_qualitative.py)) to assess semiconductor structural restoration.

---

## Quantitative Metrics

### 1. Peak Signal-to-Noise Ratio (PSNR)
Implemented in [`src/metrics/psnr_ssim.py`](file:///d:/Programming/python/semicon/src/metrics/psnr_ssim.py):
$$\text{MSE} = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W (\hat{Y}_{i,j} - Y_{i,j})^2$$
$$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{data\_range}^2}{\text{MSE}} \right)$$
- **Data Range**: Strictly set to `data_range = 1.0` reflecting the normalized floating-point dynamic range of ground-truth SEM micrographs.

### 2. Structural Similarity Index Measure (SSIM)
Implemented in [`src/metrics/psnr_ssim.py`](file:///d:/Programming/python/semicon/src/metrics/psnr_ssim.py):
$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- **Parameters**: 2D Gaussian window size $11 \times 11$, $\sigma = 1.5$, stability constants $C_1 = (0.01 \cdot 1.0)^2 = 10^{-4}$, $C_2 = (0.03 \cdot 1.0)^2 = 9 \times 10^{-4}$.

### 3. Learned Perceptual Image Patch Similarity (LPIPS)
Implemented in [`src/metrics/lpips.py`](file:///d:/Programming/python/semicon/src/metrics/lpips.py):
- **Backbone**: Pretrained AlexNet feature extractor.
- **Channel Mapping**: Since LPIPS requires 3-channel RGB inputs, single-channel SEM tensors `(B, 1, H, W)` are replicated across 3 channels `(B, 3, H, W)`: `x.repeat(1, 3, 1, 1)`.
- **Dynamic Range Mapping**: Normalized from $[0.0, 1.0]$ to $[-1.0, 1.0]$ via $x_{\text{lpips}} = 2x - 1$.

---

## Qualitative Evaluation & Visual Residual Maps

To inspect edge integrity and defect structure, [`scripts/evaluate_qualitative.py`](file:///d:/Programming/python/semicon/scripts/evaluate_qualitative.py) generates publication-grade 4-column comparison grids:
1. **Low-Dose Input**: Raw degraded $128 \times 128$ micrograph.
2. **Model Output**: $2\times$ super-resolved and denoised $256 \times 256$ micrograph.
3. **Ground Truth**: High-dose reference $256 \times 256$ micrograph.
4. **Absolute Residual Map**: Visualizing pixel-wise error magnitude $|I_{\text{pred}} - I_{\text{gt}}|$.

### Sample Visual Results

| Sample ID | Full 4-Column Comparison Grid | Fine Feature Zoom Crop |
| :---: | :---: | :---: |
| **000214** | ![Grid 000214](../results/images/qualitative_analysis/000214_comparison_grid.png) | ![Zoom 000214](../results/images/qualitative_analysis/000214_zoom_crop.png) |
| **000897** | ![Grid 000897](../results/images/qualitative_analysis/000897_comparison_grid.png) | ![Zoom 000897](../results/images/qualitative_analysis/000897_zoom_crop.png) |
| **002538** | ![Grid 002538](../results/images/qualitative_analysis/002538_comparison_grid.png) | ![Zoom 002538](../results/images/qualitative_analysis/002538_zoom_crop.png) |

---

## Evaluation Status & Technical Transparency

> [!IMPORTANT]
> - **Metric Calculation Engine**: Fully implemented, unit-tested, and verified (`tests/test_metrics.py`, `tests/test_evaluator.py`, `tests/test_qualitative_evaluator.py`).
> - **Validation Metrics on `dataset/val_split.json`**: **NOT VERIFIED**. Because no trained model checkpoint is committed in the repository, held-out validation metrics on the frozen split have not been calculated.
> - **Historical Metrics**: Reported values from experiments (`exp001`, `exp002`, `exp003`, `exp009`, `exp011`) are pre-split historical logs and must not be confused with formal held-out validation on `val_split.json`.

---

## Code References

- PSNR / SSIM Module: [`src/metrics/psnr_ssim.py`](file:///d:/Programming/python/semicon/src/metrics/psnr_ssim.py)
- LPIPS Module: [`src/metrics/lpips.py`](file:///d:/Programming/python/semicon/src/metrics/lpips.py)
- Evaluation Engine: [`src/engine/evaluator.py`](file:///d:/Programming/python/semicon/src/engine/evaluator.py)
- Qualitative Script: [`scripts/evaluate_qualitative.py`](file:///d:/Programming/python/semicon/scripts/evaluate_qualitative.py)
