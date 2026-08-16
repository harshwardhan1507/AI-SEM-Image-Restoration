# Supervised Training & Optimization Pipeline

## Overview

The training infrastructure is implemented in [`src/engine/trainer.py`](file:///d:/Programming/python/semicon/src/engine/trainer.py) and executed via [`train.py`](file:///d:/Programming/python/semicon/train.py). It provides a modular, reproducible PyTorch supervised regression framework supporting Automatic Mixed Precision (AMP), gradient clipping, numerical stability guards, and multi-objective loss combinations.

---

## Training Loop Architecture

![Supervised Training Pipeline](<../assets/diagrams/451a31c0-6670-4eb5-a588-dcf1393f92f0 - Copy.png>)

*Figure 1: Supervised training loop featuring AdamW optimization, Cosine Annealing learning rate schedule, mixed-precision GradScaler, and validation checkpointing.*

---

## Optimization Infrastructure

### 1. Optimizer Configuration
- **Optimizer**: AdamW
- **Initial Learning Rate**: $1.0 \times 10^{-3}$
- **Weight Decay**: $1.0 \times 10^{-4}$
- **Betas**: $(\beta_1 = 0.9, \beta_2 = 0.999)$

### 2. Learning Rate Schedule
- **Scheduler**: `torch.optim.lr_scheduler.CosineAnnealingLR`
- **Total Epochs**: 50 (or 30 in fast experiment cycles)
- **Minimum Learning Rate ($\eta_{\min}$)**: $1.0 \times 10^{-6}$
- Smooth monotonic decay avoids step-function gradient shocks and encourages fine convergence near flat local minima.

### 3. Mixed Precision & Hardware Acceleration
- **CUDA AMP**: `torch.amp.autocast('cuda', dtype=torch.float16)` combined with `torch.cuda.amp.GradScaler` when an NVIDIA GPU is available.
- **CPU Fallback**: Automatic device detection seamlessly falls back to standard FP32 CPU execution with `autocast('cpu', enabled=False)` when no accelerator is present.

### 4. Numerical Safety & Stability Guards
- **Finite-Loss Guard**: Before backpropagation, `torch.isfinite(loss)` validates that no NaN or Inf values have contaminated the batch loss.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents exploding gradients during initial warm-up.
- **Finite-Gradient Guard**: `GradScaler.step()` detects unscale overflows and safely skips problematic optimizer steps without corrupting model weights.

---

## Loss Functions

### 1. Differentiable Charbonnier Loss (Primary Objective)
Implemented in [`src/losses/charbonnier.py`](file:///d:/Programming/python/semicon/src/losses/charbonnier.py):
$$\mathcal{L}_{\text{Charbonnier}}(\hat{Y}, Y) = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(\hat{y}_i - y_i)^2 + \epsilon^2} \quad (\epsilon = 10^{-3})$$
The Charbonnier penalty provides a smooth, differentiable approximation of the $L_1$ norm. Unlike $L_2$ (MSE), it does not over-penalize large isolated pixel residuals, preventing over-smoothing along high-contrast semiconductor edges.

### 2. Differentiable SSIM Loss
Implemented in [`src/losses/composite.py`](file:///d:/Programming/python/semicon/src/losses/composite.py):
$$\mathcal{L}_{\text{SSIM}}(\hat{Y}, Y) = 1 - \text{SSIM}(\hat{Y}, Y)$$
Uses an $11 \times 11$ Gaussian kernel ($\sigma=1.5$) to penalize localized structural distortions and contrast discrepancies.

### 3. Frequency-Domain FFT-L1 Loss
Implemented in [`src/losses/composite.py`](file:///d:/Programming/python/semicon/src/losses/composite.py):
$$\mathcal{L}_{\text{FFT}}(\hat{Y}, Y) = \frac{1}{N} \|\; |\mathcal{F}(\hat{Y})| - |\mathcal{F}(Y)|\; \|_1$$
Computes the $L_1$ difference between the 2D Fast Fourier Transform magnitudes of prediction and target to encourage high-frequency texture recovery.

### 4. Composite Multi-Objective Loss
$$\mathcal{L}_{\text{Composite}} = w_{\text{charb}} \mathcal{L}_{\text{Charbonnier}} + w_{\text{ssim}} \mathcal{L}_{\text{SSIM}} + w_{\text{fft}} \mathcal{L}_{\text{FFT}}$$
Configured in `exp011`:
- $w_{\text{charb}} = 1.0$
- $w_{\text{ssim}} = 0.2$
- $w_{\text{fft}} = 0.05$

---

## Critical Experiment Finding: FFT-L1 Normalization

During experiment `exp011`, finetuning with Composite Loss yielded a validation PSNR of **28.53 dB** compared to **29.15 dB** for pure Charbonnier loss in `exp009`. 

A detailed code investigation revealed the mathematical root cause:
- The FFT loss in `src/losses/composite.py` computes:
  ```python
  fft_pred = torch.fft.fft2(pred, norm="backward")
  fft_target = torch.fft.fft2(target, norm="backward")
  return F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))
  ```
- `norm="backward"` is an unnormalized 2D discrete Fourier transform. The magnitude of the DFT scales with the spatial dimensions: $H \times W = 256 \times 256 = 65,536$.
- Because unnormalized frequency coefficients are orders of magnitude larger than spatial domain pixel values $[0, 1]$, the frequency term dominated the gradient updates, pulling optimization away from pixel-accurate structural alignment.

**Solution for future iterations**: Use `norm="ortho"` in `torch.fft.fft2` or scale $w_{\text{fft}}$ by $1/(HW)$.

---

## Code References

- Training Engine: [`src/engine/trainer.py`](file:///d:/Programming/python/semicon/src/engine/trainer.py)
- Main CLI Entrypoint: [`train.py`](file:///d:/Programming/python/semicon/train.py)
- Loss Functions: [`src/losses/charbonnier.py`](file:///d:/Programming/python/semicon/src/losses/charbonnier.py), [`src/losses/composite.py`](file:///d:/Programming/python/semicon/src/losses/composite.py), [`src/losses/psnr_loss.py`](file:///d:/Programming/python/semicon/src/losses/psnr_loss.py)
- Checkpointing: [`src/engine/checkpoint.py`](file:///d:/Programming/python/semicon/src/engine/checkpoint.py)
