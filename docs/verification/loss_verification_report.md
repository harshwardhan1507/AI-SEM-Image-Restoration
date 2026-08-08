# Verification Report: Differentiable Restoration Losses

**Feature**: Issue #8 — Differentiable Restoration Loss Functions  
**Branch**: `8-featlosses-implement-charbonnier-and-psnr-loss-functions`  

---

## 1. Objective & Scope

Implemented differentiable restoration loss modules (`CharbonnierLoss` and `PSNRLoss`) and a configuration-driven loss builder (`build_loss`) in `src/losses/`. All loss calculations strictly use PyTorch autograd tensor operations without converting to NumPy or relying on external metric libraries.

---

## 2. API Specification & Mathematical Definitions

### Charbonnier Loss (`CharbonnierLoss`)
- **API Signature**:
  ```python
  CharbonnierLoss(eps: float = 1e-3, reduction: str = "mean")
  ```
- **Mathematical Formula**:
  $$L_{\text{Charbonnier}}(P, T) = \text{reduction}\left( \sqrt{(P - T)^2 + \epsilon^2} \right)$$
- **Documented Rationale**:
  $\epsilon = 1e-3$ (DOCUMENTED FACT in `docs/project_roadmap.md` line 254). Provides smooth differentiable approximation to L1 loss and prevents zero-gradients when $P = T$.

### Differentiable PSNR Loss (`PSNRLoss`)
- **API Signature**:
  ```python
  PSNRLoss(data_range: float = 1.0, eps: float = 1e-8, reduction: str = "mean")
  ```
- **Mathematical Formula**:
  $$\text{MSE}_b = \text{mean}_{C,H,W}\left( (P_b - T_b)^2 \right)$$
  $$L_b = \frac{10}{\ln 10} \cdot \ln \left( \frac{\text{MSE}_b + \epsilon}{D^2} \right)$$
  $$L = \text{reduction}(L_b)$$
- **Per-Sample Batch Semantics**:
  For batched inputs $(B, C, H, W)$, MSE is computed per-sample across non-batch dimensions. `reduction="none"` returns per-sample loss tensor of shape $(B,)$, matching evaluation semantics in Issue #7.
- **Numerical Stabilization Rationale**:
  When $P = T$, $\text{MSE}_b = 0$. Using $\text{MSE}_b + 1e-8$ yields a finite scalar loss of $-80.0$ dB for $D=1.0$ and finite autograd gradients $\frac{\partial L}{\partial P} = 0$ (ENGINEERING ASSUMPTION & TESTED BEHAVIOR).

---

## 3. Classification of Facts, Assumptions & Test Results

| Aspect | Status | Details |
| :--- | :--- | :--- |
| **Charbonnier Epsilon ($1e-3$)** | **DOCUMENTED FACT** | Explicitly specified in `docs/project_roadmap.md` line 254 |
| **PSNR Data Range ($1.0$)** | **DOCUMENTED FACT** | Confirmed by dataset clip range `[0, 1]` in `sem_dataset.py` and `psnr_ssim.py` |
| **PSNR Stabilization Epsilon ($1e-8$)** | **ENGINEERING ASSUMPTION** | Chosen to guarantee finite loss ($-80.0$ dB at zero MSE) and finite gradients |
| **Per-Sample Batch Semantics** | **ENGINEERING ASSUMPTION** | Aligned with Issue #7 metric evaluation semantics |
| **Autograd Gradient Finiteness** | **TEST RESULT** | Verified: `pred.grad is not None` and `torch.isfinite(pred.grad).all() == True` |
| **Full Repository Regression** | **TEST RESULT** | Verified: 137 passed, 3 skipped, 0 failures |

---

## 4. Empirical Quality Gate & Test Execution Summary

### Test Results (`tests/test_losses.py`)
```text
tests/test_losses.py::test_charbonnier_construction PASSED
tests/test_losses.py::test_charbonnier_known_values PASSED
tests/test_losses.py::test_charbonnier_zero_residual PASSED
tests/test_losses.py::test_charbonnier_reductions PASSED
tests/test_losses.py::test_charbonnier_autograd_and_non_mutation PASSED
tests/test_losses.py::test_charbonnier_shape_mismatch PASSED
tests/test_losses.py::test_psnr_loss_construction PASSED
tests/test_losses.py::test_psnr_loss_known_values PASSED
tests/test_losses.py::test_psnr_loss_batch_semantics PASSED
tests/test_losses.py::test_psnr_loss_zero_mse_stability PASSED
tests/test_losses.py::test_psnr_loss_autograd_finiteness PASSED
tests/test_losses.py::test_psnr_loss_shape_mismatch PASSED
tests/test_losses.py::test_build_loss_charbonnier PASSED
tests/test_losses.py::test_build_loss_psnr PASSED
tests/test_losses.py::test_build_loss_unsupported PASSED

============================= 15 passed in 1.99s ==============================
```

### Code Style & Quality Gates
- `black src/losses tests/test_losses.py`: Passed (100% formatted)
- `isort src/losses tests/test_losses.py`: Passed
- `ruff check src/losses tests/test_losses.py`: Passed (0 lint errors)
- `pytest` (full test suite): `137 passed, 3 skipped in 23.72s` (0 failures)
