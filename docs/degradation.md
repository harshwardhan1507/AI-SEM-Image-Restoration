# Physics-Inspired SEM Degradation Model

## Overview

In Critical Dimension Scanning Electron Microscopy (CD-SEM), reducing the primary beam current and exposure dwell time is necessary to prevent photoresist shrinkage, pattern distortion, and wafer charging. However, lower electron dosage degrades image quality through severe quantum shot noise, surface emission fluctuations, and coarse beam stepping.

This repository implements a **physics-inspired synthetic degradation pipeline** in [`src/datasets/transforms.py`](file:///d:/Programming/python/semicon/src/datasets/transforms.py) to simulate real-world SEM degradation mechanisms from high-resolution clean ground-truth arrays.

---

## Degradation Formulation

```
                     Clean High-Resolution Ground Truth (256×256)
                                         │
                                         ▼
                      [Optional Gaussian Blur (p=0.15)]
                                         │
                                         ▼
                 [OpenCV INTER_CUBIC 2× Downsampling → 128×128]
                                         │
                                         ▼
             [Multiplicative Gamma Speckle Noise (p=1.0, σ ∈ [0.040, 0.231])]
                                         │
                                         ▼
             [Poisson Shot Noise with Tail Preservation (p=1.0, S ∈ [110, 180])]
                                         │
                                         ▼
             [Optional Additive Gaussian Noise (p=0.0 in exp009, OOD Insurance)]
                                         │
                                         ▼
                    Synthesized Degraded Micrograph (128×128)
```

---

## Physical Degradation Mechanisms

### 1. Spatial Resolution Loss ($2\times$ Downsampling)
Coarse raster scanning during high-throughput wafer inspection reduces spatial resolution by $2\times$. The pipeline uses OpenCV `cv2.INTER_CUBIC` downsampling without anti-aliasing filtering, faithfully mirroring the aliasing and spatial quantization of physical electron detectors.

### 2. Multiplicative Gamma Speckle Noise
Secondary electron (SE) emission varies across nanometer-scale grain boundaries and material work functions. This is modeled as multiplicative Gamma noise:
$$I_{\text{speckle}} = I_{\text{clean}} \odot \eta_{\gamma}, \quad \eta_{\gamma} \sim \text{Gamma}\left(L, \frac{1}{L}\right), \quad L = \frac{1}{\sigma_s^2}$$
where $\sigma_s \in [0.040, 0.231]$ controls speckle intensity.

### 3. Signal-Dependent Poisson Shot Noise
Electron arrival at the detector follows Poisson counting statistics where noise variance is proportional to signal level ($\text{SNR} \propto \sqrt{N_{\text{PE}}}$):
$$I_{\text{poisson}} = \frac{\mathcal{P}\left(\max(I_{\text{speckle}}, 0) \cdot S\right)}{S} + \min(I_{\text{speckle}}, 0)$$
where $S \in [110.0, 180.0]$ is the photon/electron scaling factor. Negative pixel tails are preserved separately to maintain physical sensor baseline statistics.

### 4. Additive Gaussian Thermal Noise (Configurable OOD Insurance)
Thermal fluctuations in the detector electronics produce additive Gaussian noise:
$$I_{\text{final}} = I_{\text{poisson}} + \mathcal{N}(0, \sigma_g^2)$$
In the primary submission experiment (`exp009`), Gaussian noise is set to $p=0.0$ because Poisson-Gamma statistics accurately model the in-distribution SEM data. Additive Gaussian noise is retained as a configurable module to provide robustness against out-of-distribution (OOD) test datasets.

---

## Implemented Parameter Configuration (`exp009`)

Hyperparameters from [`configs/experiments/exp009_augmentation_poisson.yaml`](file:///d:/Programming/python/semicon/configs/experiments/exp009_augmentation_poisson.yaml):

| Parameter | Value | Description |
| :--- | :---: | :--- |
| `synthetic_degradation.probability` | `0.5` | Probability of applying synthetic degradation vs raw paired input during training. |
| `blur_prob` | `0.15` | Probability of pre-downsampling Gaussian blur. |
| `speckle_prob` | `1.0` | Probability of applying multiplicative Gamma speckle noise. |
| `speckle_sigma_range` | `[0.040, 0.231]` | Uniform standard deviation sampling range for Gamma noise. |
| `poisson_prob` | `1.0` | Probability of applying Poisson shot noise. |
| `poisson_scale_range` | `[110.0, 180.0]` | Scaling parameter range for Poisson counting process. |
| `gaussian_prob` | `0.0` | Probability of additive Gaussian noise (inactive in exp009; available for OOD). |
| `gaussian_sigma_range` | `[0.0, 0.0183]` | Standard deviation range for additive Gaussian noise. |

---

## Code References

- Synthetic Degradation Implementation: [`src/datasets/transforms.py`](file:///d:/Programming/python/semicon/src/datasets/transforms.py) (`SyntheticDegradation`, `PairedTransforms`)
- Verification Script: [`scripts/verify_degradation.py`](file:///d:/Programming/python/semicon/scripts/verify_degradation.py)
- Degradation Analysis Script: [`scripts/measure_real_stats.py`](file:///d:/Programming/python/semicon/scripts/measure_real_stats.py)
