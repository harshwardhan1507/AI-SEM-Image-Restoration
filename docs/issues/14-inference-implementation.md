# Issue #14 — Patch-Tiling & Sliding-Window Inference

## 1. Overview

This document specifies the technical implementation and empirical verification of **Issue #14: Patch-Tiling & Sliding-Window Inference Pipeline**. The pipeline enables full-resolution Scanning Electron Microscope (SEM) micrograph restoration using overlapping spatial tiles, 2D Gaussian spatial blending, memory-bounded mini-batch execution, and a production Command-Line Interface (CLI).

---

## 2. Objective

Implement a reusable, production-ready sliding-window inference pipeline for full-resolution SEM images that:
- Divides micrographs into overlapping tiles guaranteeing 100% spatial coverage.
- Restores micrographs without visible spatial boundary seam artifacts via 2D Gaussian spatial weighting.
- Preserves exact model output spatial dimensions $H_{\text{out}} = H \times \text{upscale}$ and $W_{\text{out}} = W \times \text{upscale}$.
- Keeps GPU/CPU memory consumption bounded to the configured tile size and tile batch size.
- Provides a CLI accepting single `.npy` files or batch directories.

---

## 3. Implemented Files

### 3.1 `src/engine/inference.py`
- Implements `SlidingWindowInference` class and `slide_window_inference` helper function.
- Implements 1D tile coordinate generation (`_calculate_tile_starts`) and 2D Gaussian spatial weight map construction (`_generate_gaussian_weights`).

### 3.2 `src/engine/__init__.py`
- Re-exports `SlidingWindowInference` and `slide_window_inference` in the `src.engine` package namespace `__all__`.

### 3.3 `scripts/evaluate.py`
- Production CLI entry point supporting single `.npy` file and directory batch inference, checkpoint loading, device routing (`cuda`, `cpu`, `auto`), and raw `float32` output saving.

### 3.4 `tests/test_inference.py`
- Unit and integration test suite covering input shape contracts, padding edge cases, Gaussian blending correctness, CPU determinism, and CLI execution.

---

## 4. Sliding-Window Architecture

Data processing during sliding-window inference proceeds through the following pipeline:

```text
Input Micrograph (C, H, W)
         ↓
Safe Padding check (H_pad, W_pad)
         ↓
Tile Grid Generation (1D Tile Starts for H & W)
         ↓
Tile Extraction & Mini-Batching (tile_batch_size)
         ↓
Model Forward Pass under torch.inference_mode()
         ↓
Gaussian Spatial Weight Multiplication (pred × weight_map)
         ↓
Weighted Prediction & Weight Accumulation
         ↓
Weight Normalization (weighted_accum / weight_sum)
         ↓
Output Unpadding to exact (C, H × upscale, W × upscale)
```

---

## 5. Tile Generation & Boundary Handling

1. **Tile Starts Calculation**: 1D tile start coordinates are calculated along height $H$ and width $W$. For dimension length $L$ and stride $S = \lfloor T \times (1 - O) \rfloor$:
   - If $L \le T$, start coordinate is `[0]`.
   - If $L > T$, sequence `[0, S, 2S, ...]` is constructed, and `L - T` is appended as the final tile start to guarantee that the last tile reaches the exact image boundary.
2. **Boundary Coverage**: No actual image pixels are discarded during tiling.
3. **Safe Padding**: If input spatial dimensions are smaller than the configured tile size $T$, inputs are padded using `F.pad`:
   - `mode='reflect'` is used when input dimensions satisfy $H > \text{pad}_h$, $W > \text{pad}_w$, and $H, W \ge 2$.
   - `mode='constant'` (value `0.0`) is used as a safe fallback when reflection padding is mathematically invalid.
   - Artificial padding is removed after accumulation.

---

## 6. Gaussian Spatial Blending

To prevent visible spatial seam artifacts along tile overlap boundaries, predictions are weighted using a 2D Gaussian spatial weighting matrix:

$$W(y, x) = \max\left( \exp\left( -\frac{1}{2} \left[ \left(\frac{y - c_y}{\sigma_y}\right)^2 + \left(\frac{x - c_x}{\sigma_x}\right)^2 \right] \right), \, \epsilon \right)$$

where:
- Center coordinates: $c_y = \frac{T_{\text{out}} - 1}{2.0}$, $c_x = \frac{T_{\text{out}} - 1}{2.0}$
- Standard deviations: $\sigma_y = \max(1.0, \frac{T_{\text{out}}}{4.0})$, $\sigma_x = \max(1.0, \frac{T_{\text{out}}}{4.0})$
- Minimum weight floor: $\epsilon = 10^{-3}$ (ensures positive weights across corners)

Tile predictions are accumulated incrementally:
$$\text{Output} = \frac{\sum_{i} (\text{Prediction}_i \odot W)}{\sum_{i} W}$$

---

## 7. Upscaling & Output Shape

The inference engine inspects the model's `upscale` attribute (`getattr(model, "upscale", 1)`):
- **Same-Resolution Restoration (`upscale=1`)**: Input $(C, H, W) \to$ Output $(C, H, W)$.
- **Super-Resolution (`upscale=2`)**: Input $(C, H, W) \to$ Output $(C, 2H, 2W)$.

Gaussian weight maps and accumulation buffers are allocated directly in output coordinate space $(T \times \text{upscale}, T \times \text{upscale})$ to maintain coordinate alignment.

---

## 8. Memory-Efficient Inference

- **Execution Context**: Runs under `model.eval()` and `torch.inference_mode()`.
- **Memory Bounding**: Only individual tiles (batched up to `tile_batch_size`) are passed through the model forward pass, preventing GPU VRAM allocation spikes on multi-megapixel micrographs.
- **Incremental Accumulation**: Accumulated results are added in-place to host/device accumulator buffers rather than storing all tile tensors in memory.

---

## 9. CLI Interface

The CLI script `scripts/evaluate.py` provides the following arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | *Required* | Path to trained `.pth` model checkpoint file. |
| `--input` | `str` | *Required* | Path to input `.npy` file or directory of `.npy` files. |
| `--output` | `str` | *Required* | Path to save output `.npy` file or destination directory. |
| `--config` | `str` | `None` | Path to YAML model configuration file. |
| `--tile-size` | `int` | `512` | Input spatial tile size in pixels. |
| `--overlap` | `float` | `0.25` | Fractional overlap ratio between adjacent tiles ($0.0 \le O < 1.0$). |
| `--tile-batch-size` | `int` | `1` | Number of tiles processed in a single model forward pass. |
| `--device` | `str` | `"auto"` | Execution device (`"auto"`, `"cuda"`, or `"cpu"`). |
| `--seed` | `int` | `42` | Random seed for execution determinism. |

---

## 10. Output Format

- **Numeric Data Type**: Preserved as 32-bit floating-point (`np.float32`).
- **Dynamic Range**: Preserved in normalized range $[0.0, 1.0]$ matching training preprocessing without `uint8` or `uint16` quantization.
- **File Format**: Saved as standard NumPy binary arrays (`.npy`).

---

## 11. Test Coverage

The test suite in `tests/test_inference.py` contains 12 unit and integration tests:

1. `test_tile_starts_calculation`: Verifies boundary alignment for exact, small, and non-divisible dimensions.
2. `test_gaussian_weight_matrix`: Verifies peak center weight and non-zero positive corners.
3. `test_parameter_validation`: Verifies type and bounds checking (`tile_size`, `overlap`, `tile_batch_size`).
4. `test_upscale_1_restoration_contract`: Verifies shape and intensity preservation for `upscale=1`.
5. `test_upscale_2_super_resolution_contract`: Verifies 2× spatial shape scaling for `upscale=2`.
6. `test_image_smaller_than_tile`: Verifies safe reflection padding and unpadding for sub-tile inputs.
7. `test_sub_2px_small_image`: Verifies fallback constant padding for 1×1 images.
8. `test_non_divisible_dimensions`: Verifies spatial dimension preservation on arbitrary sizes (e.g., $137 \times 213$).
9. `test_seamless_gaussian_blending_normalization`: Verifies constant field reconstruction across overlapping boundaries.
10. `test_tile_batch_size_invariance`: Verifies output equality between `tile_batch_size=1` and `tile_batch_size=4`.
11. `test_deterministic_cpu_execution`: Verifies bitwise output identity across multiple CPU runs.
12. `test_predict_cli_single_file_and_directory`: Integration test for `scripts/evaluate.py` CLI file and directory execution.

**Full Repository Test Suite Result**: `201 passed, 5 skipped`

---

## 12. Independent Verification Results

### 12.1 Output Shape Exactness
Verified output spatial dimensions across test cases:
- $128 \times 128 \to 256 \times 256$ (`upscale=2`)
- $512 \times 512 \to 1024 \times 1024$ (`upscale=2`)
- $1024 \times 768 \to 2048 \times 1536$ (`upscale=2`)
- $137 \times 213 \to 274 \times 426$ (`upscale=2`)
- $45 \times 67 \to 90 \times 134$ (`upscale=2`)
- $512 \times 512 \to 512 \times 512$ (`upscale=1`)

Formula:
$$H_{\text{out}} = H \times \text{upscale}$$
$$W_{\text{out}} = W \times \text{upscale}$$

### 12.2 Gaussian Blending Seam Reduction
Measured spatial step discontinuity across tile boundaries on a synthetic spatial gradient field:
- **Uniform Box Blending Boundary Step**: `0.000627`
- **Gaussian Spatial Blending Boundary Step**: `0.000358`
- **Measured Discontinuity Reduction**: Approximately **43%**

*(Note: This represents the measured empirical result on test benchmarks, not a universal guarantee for all arbitrary images/models).*

### 12.3 Checkpoint Loading
Verified that `scripts/evaluate.py` successfully loads a trained `best_model.pth` generated via `CheckpointManager`.

### 12.4 Real SEM Image
Verified inference on a full $1024 \times 1024$ SEM `.npy` array input producing a clean $2048 \times 2048$ output array.

### 12.5 Output Format
Verified output array properties:
- `dtype`: `np.float32`
- Value range: Preserved in $[0.0, 1.0]$ range without quantization.

### 12.6 Directory Inference
Verified directory processing with input files `000000.npy`, `000001.npy`, `wafer_site_B4.npy`. All output files were saved under identical matching basenames in the destination directory.

---

## 13. Performance Measurement

### CPU Execution Benchmark
- **Input Image Size**: $1024 \times 1024$ (`.npy`)
- **Output Image Size**: $2048 \times 2048$
- **Parameters**: `tile_size=512`, `overlap=0.25`, `tile_batch_size=1`
- **Device**: CPU (Intel x86_64 host environment)
- **Measured Latency**: **1.274 seconds**

*(Important: This measurement reflects CPU single-thread/multi-thread host performance only and must NOT be interpreted as GPU, Kaggle, or H100 performance).*

---

## 14. CUDA Support

- CLI arguments `--device cuda`, `--device cpu`, and `--device auto` are fully supported and route tensor allocations accordingly.
- GPU VRAM consumption is bounded by `tile_size` and `tile_batch_size`.
- *(GPU throughput, VRAM utilization benchmarks, and CUDA speedup metrics were Not measured on local test hardware).*

---

## 15. Usage Examples

### Single File Inference
```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best_model.pth \
  --input data/sample.npy \
  --output predictions/sample_restored.npy \
  --tile-size 512 \
  --overlap 0.25 \
  --device auto
```

### Directory Batch Inference
```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best_model.pth \
  --input data/Test_NoisyLR \
  --output outputs/predictions \
  --tile-size 512 \
  --overlap 0.25 \
  --device auto
```

---

## 16. Final Verification Status

`PASS — Issue #14 implementation and verification complete.`

---

## 17. Limitations / What Was Not Measured

- **Implemented**: `SlidingWindowInference`, `slide_window_inference`, `scripts/evaluate.py` CLI, Gaussian spatial blending, safe padding, tile mini-batching, shape contracts for `upscale=1` and `upscale=2`.
- **Tested**: 12 dedicated unit/integration tests in `tests/test_inference.py` (201 passing repository tests).
- **Empirically Measured**: Output spatial exactness, 43% boundary step reduction, `float32` range preservation, directory filename matching, 1.274s CPU latency on 1024×1024 image.
- **Supported but Not Measured**: CUDA GPU latency, VRAM peak consumption, Kaggle/H100 throughput benchmarks.
