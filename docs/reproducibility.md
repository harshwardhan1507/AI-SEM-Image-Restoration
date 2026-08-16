# Pipeline Reproducibility Guide

## Overview

This guide provides the complete, step-by-step instructions to set up the environment, verify model complexity, run test suites, profile datasets, train models from scratch, and execute inference pipelines.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- PyTorch 2.0+
- CUDA 11.8+ (Optional; CPU execution is fully supported)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/harshwardhan1507/AI-SEM-Image-Restoration.git
cd AI-SEM-Image-Restoration

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install runtime and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install repository in editable mode
pip install -e .
```

---

## 2. Model & Parameter Verification

Verify the parameter counts and layer topology across Width 32, 48, and 64:

```bash
python scripts/verify_params.py
```

Expected output:
```text
Width 32: 1,129,028 parameters
Width 48: 2,521,444 parameters
Width 64: 4,465,796 parameters
```

---

## 3. Test Suite Verification

Run the full pytest suite (248 unit and integration tests):

```bash
pytest tests/
```

Expected result:
```text
248 passed, 5 skipped in ~35s
```
*(The 5 skipped tests require optional CUDA and C++ MSVC compiler availability for `torch.compile` and CUDA-specific AMP).*

---

## 4. Dataset Profiling & Characterization

Extract statistical distributions and dynamic range profiles from the `.npy` dataset:

```bash
python scripts/analyze_dataset.py --dataset-dir dataset/
```

Benchmark DataLoader ingestion throughput:

```bash
python scripts/benchmark_dataset.py
```

---

## 5. Training Pipelines

### Primary Submission Model (`exp009` — Physics-Calibrated Baseline)
Train NAFNet Width 48 with Poisson shot noise and Gamma speckle augmentations using Charbonnier loss:

```bash
python train.py --config configs/experiments/exp009_augmentation_poisson.yaml
```

Checkpoints, TensorBoard events, and execution logs will be saved to:
- Checkpoints: `outputs/checkpoints/exp009_augmentation_poisson/`
- TensorBoard: `outputs/tensorboard/exp009_augmentation_poisson/`
- Logs: `logs/exp009_augmentation_poisson/`

### Optional Finetuning (`exp011` — Composite Loss)
Finetune an existing `exp009` checkpoint using composite multi-objective loss:

```bash
python train.py \
  --config configs/experiments/exp011_finetune_ssim.yaml \
  --finetune outputs/checkpoints/exp009_augmentation_poisson/best_model.pth
```

---

## 6. Inference Execution

Run $2\times$ super-resolution restoration on a directory of degraded `.npy` arrays:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/exp009_augmentation_poisson/best_model.pth \
  --input dataset/test/degraded/ \
  --output results/predictions/ \
  --tile-size 256 \
  --overlap 0.25
```

---

## 7. Qualitative Evaluation Grids

Generate 4-column qualitative evaluation grids and fine-feature zoom crops:

```bash
python scripts/evaluate_qualitative.py \
  --dataset-dir dataset/ \
  --split val \
  --baseline-checkpoint outputs/checkpoints/exp009_augmentation_poisson/best_model.pth \
  --improved-checkpoint outputs/checkpoints/exp011_finetune_ssim/best_model.pth \
  --include-bicubic-ref \
  --num-samples 6
```

Output figures are written to `results/images/qualitative_analysis/`.
