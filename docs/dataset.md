# Dataset Characterization & Preprocessing Pipeline

## Overview

The dataset consists of paired Scanning Electron Microscope (SEM) micrographs stored as **32-bit floating-point NumPy arrays (`.npy`)**. Using native `.npy` float32 storage prevents lossy 8-bit dynamic range truncation, preserving the physical sensor signal and electron emission characteristics.

---

## Dataset Ingestion & Preprocessing Workflow

![Dataset Ingestion Pipeline](<../assets/diagrams/75fd0087-52d4-48b6-b54b-0d7fc7d3c5ee - Copy.png>)

*Figure 1: Dataset ingestion, intensity handling, spatial augmentation, and memory-mapped DataLoader collation.*

---

## Data Representations & Array Dimensions

| Stream | Dimensions | Format | Dynamic Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| **NoisyLR (Input)** | $128 \times 128$ | `float32` | Raw (Unclipped) | Degraded low-dose SEM micrograph with Poisson shot noise, multiplicative speckle, and $2\times$ spatial downsampling. |
| **Ground Truth (Target)** | $256 \times 256$ | `float32` | Clipped $[0.0, 1.0]$ | High-dose reference SEM micrograph with clean structural boundaries and high signal-to-noise ratio (SNR). |

---

## Intensity Handling Policy

A critical engineering decision in this pipeline is the differential handling of input and target pixel intensities:

1. **NoisyLR Inputs (Unclipped)**:
   - Ingested via `SEMDataset` with `clip=False`.
   - Raw floating-point values are intentionally **not clipped** to $[0.0, 1.0]$.
   - Sensor noise, negative baseline undershoots (e.g. down to $-0.002$), and high-energy detector spikes ($> 1.0$) are preserved so that the model learns the true physical noise distribution.

2. **Ground Truth Targets (Clipped)**:
   - Ingested via `SEMDataset` with `clip=True`.
   - Bounded strictly within $[0.0, 1.0]$ via `np.clip(arr, 0.0, 1.0)`.
   - Ensures the supervised regression targets represent normalized physical reflectance/emission values without detector artifact anomalies.

```python
# src/datasets/sem_dataset.py snippet
def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, Optional[torch.Tensor]]]:
    pair: DatasetPair = self.pairs[index]
    # Do not clip the input NoisyLR (preserves physical noise profile)
    input_tensor = self._process_array(pair.input_path, clip=False)

    target_tensor: Optional[torch.Tensor] = None
    if pair.target_path is not None:
        # GT target is clipped strictly to [0, 1]
        target_tensor = self._process_array(pair.target_path, clip=True)
```

---

## Frozen Dataset Splits

To prevent data leakage and guarantee experimental reproducibility, dataset splits are fixed via JSON index files:

- **Total Paired Samples**: 3,200 samples
- **Training Split (`dataset/train_split.json`)**: 2,882 samples (~90%)
- **Validation Split (`dataset/val_split.json`)**: 318 samples (~10%)

Both split files contain exact array filename lists (e.g., `000001.npy`, `000002.npy`, ...), ensuring identical sample partitioning across all experiments.

---

## Spatial Augmentations

Geometric augmentations are implemented via `albumentations` in [`src/datasets/transforms.py`](file:///d:/Programming/python/semicon/src/datasets/transforms.py):

- **Horizontal Flip**: $p = 0.5$
- **Vertical Flip**: $p = 0.5$
- **Random 90° Rotation**: $p = 0.5$

Transformations are applied synchronously to paired input `(128, 128)` and target `(256, 256)` arrays using `additional_targets={"target": "image"}`, maintaining strict spatial alignment across scales.

---

## Loader Throughput & Performance Benchmarks

Dataset ingestion was benchmarked across 3,200 paired `.npy` files (`results/benchmarks/dataset_benchmark_report.md`):

| Performance Metric | Measured Value | Standard Target | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Peak Init Memory** | `3.608 MB` | $< 10.0 \text{ MB}$ | **PASS** |
| **P95 Fetch Latency** | `3.546 ms` | $< 5.0 \text{ ms}$ | **PASS** |
| **100-Sample Read Time** | `0.297 s` | $< 0.5 \text{ s}$ | **PASS** |
| **Sequential Throughput** | `336.6 samples/sec` | Stream-ready | **PASS** |
| **Tensor Data Type** | `torch.float32` | Bounded $[0.0, 1.0]$ | **PASS** |

---

## Visual Dataset Analysis

### Sample Image Pairs
![Sample SEM Image Pairs](../results/images/dataset_analysis/sample_image_pairs_comparison.png)

*Figure 2: Paired SEM micrographs comparing low-dose degraded input ($128 \times 128$) with high-dose ground truth reference ($256 \times 256$).*

### Pixel Intensity Distribution
![Pixel Intensity Histogram](../results/images/dataset_analysis/pixel_intensity_histogram.png)

*Figure 3: Floating-point pixel intensity distributions across dataset samples.*

---

## Code References

- Dataset Loader: [`src/datasets/sem_dataset.py`](file:///d:/Programming/python/semicon/src/datasets/sem_dataset.py)
- Dataset Scanner: [`src/datasets/scanner.py`](file:///d:/Programming/python/semicon/src/datasets/scanner.py)
- Data Validator: [`src/datasets/validator.py`](file:///d:/Programming/python/semicon/src/datasets/validator.py)
- Augmentations & Transforms: [`src/datasets/transforms.py`](file:///d:/Programming/python/semicon/src/datasets/transforms.py)
- DataLoader Builder: [`src/datasets/builder.py`](file:///d:/Programming/python/semicon/src/datasets/builder.py)
- Benchmark Script: [`scripts/benchmark_dataset.py`](file:///d:/Programming/python/semicon/scripts/benchmark_dataset.py)
