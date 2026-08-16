# Production Inference & Sliding-Window Pipeline

## Overview

The inference infrastructure is implemented in [`src/engine/inference.py`](file:///d:/Programming/python/semicon/src/engine/inference.py) and exposed via [`scripts/evaluate.py`](file:///d:/Programming/python/semicon/scripts/evaluate.py). It supports both **direct full-image restoration** and **memory-bounded sliding-window inference with Gaussian blending** for large SEM inspection fields.

---

## Inference Pipeline Architecture

![Production Inference Pipeline](<../assets/diagrams/a474e8b1-4854-42e0-97f2-16cc47acabbb - Copy.png>)

*Figure 1: Memory-bounded sliding-window inference engine with 2D Gaussian spatial weighting to eliminate tiling seams.*

---

## Inference Modes

### 1. Direct Whole-Image Inference
For standard sized micrographs ($128 \times 128 \to 256 \times 256$):
1. Ingests single-channel float32 `.npy` arrays.
2. Applies reflection padding to the nearest multiple of $2^L = 8$.
3. Executes a single forward pass under `torch.inference_mode()`.
4. Crops out pad boundaries to return exact $2\times$ spatial resolution.

### 2. Sliding-Window Tiling with Gaussian Blending
For arbitrarily large inspection fields (e.g. $1024 \times 1024$ or full wafer scans):
1. **Overlapping Patch Tiling**: Slices input into spatial patches of size $P \times P$ (e.g., $256 \times 256$) with configurable overlap $O = 0.25$ and stride $S = \lfloor P \cdot (1 - O) \rfloor$.
2. **Batched Forward Pass**: Feeds patch mini-batches to NAFNet under `torch.inference_mode()`.
3. **2D Gaussian Spatial Weighting**: Weights each restored tile by a 2D Gaussian kernel centered on the patch:
   $$W(y, x) = \max\left( \exp\left( -\frac{1}{2} \left[ \frac{(y - c_y)^2}{\sigma_y^2} + \frac{(x - c_x)^2}{\sigma_x^2} \right] \right), 10^{-3} \right)$$
   where $\sigma = P_{\text{out}} / 4$.
4. **Accumulation & Normalization**: Reconstructs the full field by dividing accumulated weighted pixels by the total spatial weight:
   $$\hat{I}_{\text{full}}(y, x) = \frac{\sum_i (P_i(y, x) \cdot W_i(y, x))}{\sum_i W_i(y, x)}$$
   This guarantees zero visible grid boundary artifacts across patch seams.

---

## Checkpoint Architecture Auto-Detection

The inference engine dynamically inspects the loaded checkpoint `state_dict` to automatically determine:
- Model width ($C=32, 48, 64$) from the first layer convolution channel count.
- Upscale factor ($2\times$) from the PixelShuffle output layer projection.
- Block depth per stage.

This avoids manual architectural flag mismatches when running evaluation across diverse checkpoints.

---

## CLI Usage

### Evaluate on Single File or Directory
```bash
# Run inference on a directory of degraded .npy arrays:
python scripts/evaluate.py \
  --checkpoint path/to/trained_checkpoint.pth \
  --input path/to/noisy_arrays/ \
  --output results/predictions/ \
  --tile-size 256 \
  --overlap 0.25
```

> [!IMPORTANT]
> **Checkpoint Availability Notice**:
> The inference script and pipeline logic are fully implemented and verified via unit tests (`tests/test_inference.py`). However, **no final trained model checkpoint is stored in this repository**. To execute inference, a checkpoint must first be trained via `train.py`.

---

## Code References

- Inference Engine: [`src/engine/inference.py`](file:///d:/Programming/python/semicon/src/engine/inference.py)
- CLI Evaluation Script: [`scripts/evaluate.py`](file:///d:/Programming/python/semicon/scripts/evaluate.py)
- Inference Unit Tests: [`tests/test_inference.py`](file:///d:/Programming/python/semicon/tests/test_inference.py)
