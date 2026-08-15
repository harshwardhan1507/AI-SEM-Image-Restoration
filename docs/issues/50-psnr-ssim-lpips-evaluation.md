# PSNR, SSIM and LPIPS for SEM Restoration Evaluation

## Objective

Review the scientific foundations, interpretation, and limitations of PSNR, SSIM, and LPIPS for evaluating image restoration, with particular attention to SEM and microscopy imagery.

The project evaluates restored low-resolution SEM images against clean high-resolution ground truth.

---

## 1. PSNR

### What it measures

PSNR measures pixel-wise reconstruction fidelity through the Mean Squared Error between the restored image and the ground-truth reference.

Higher PSNR indicates lower pixel-wise reconstruction error.

### Strengths

- Simple and mathematically well-defined.
- Widely used in image restoration and super-resolution benchmarks.
- Provides a direct measure of pixel-level fidelity.
- Closely related to MSE, making it useful when exact reconstruction accuracy is important.

### Limitations

PSNR does not directly model human visual perception.

Small spatial shifts can produce large PSNR penalties even when the images remain visually or structurally similar.

Pixel-wise metrics can also favor solutions that reduce numerical error while losing perceptually important fine details.

### SEM relevance

PSNR is useful for measuring how closely a restored SEM image matches its reference at the pixel level.

However, SEM-specific research demonstrates that high numerical scores do not necessarily guarantee preservation of geometric structures.

### Evidence Classification

**Direct Literature Evidence**

---

## 2. SSIM

### What it measures

SSIM evaluates image similarity using three major components:

- Luminance
- Contrast
- Structure

Unlike PSNR, SSIM considers local relationships between pixels rather than treating every pixel independently.

Higher SSIM indicates greater structural similarity.

### Strengths

- Captures local structural information.
- Provides information beyond pure pixel-wise error.
- More closely reflects structural characteristics important for image interpretation than MSE alone.
- Useful for detecting some forms of structural degradation that pixel-wise metrics may not represent well.

### Limitations

SSIM has documented limitations with uniform luminance and contrast changes.

Its behavior can also depend on the scale and parameters used when computing local statistics.

Single-scale SSIM may not adequately represent all perceptual differences across different spatial scales.

### SEM relevance

Structural preservation is particularly important for SEM imagery containing:

- Circular structures
- Contact holes
- Fine edges
- Repetitive patterns
- Small geometric features

SSIM therefore provides information that complements PSNR, but it should not be treated as a complete measure of SEM restoration quality.

### Evidence Classification

**Direct Literature Evidence**

---

## 3. LPIPS

### What it measures

LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual distance using feature representations extracted from deep neural networks.

Lower LPIPS indicates greater perceptual similarity.

Unlike PSNR and SSIM, LPIPS is designed around learned feature representations rather than direct pixel-wise comparison.

### Strengths

- Captures perceptual differences that may not be reflected by pixel-wise metrics.
- Can provide information about texture and visual similarity.
- Has been used in image restoration and SEM research alongside traditional metrics.
- Can help identify cases where numerical pixel accuracy does not correspond to visual quality.

### Limitations

Perceptual feature similarity does not guarantee physical correctness.

A restoration can appear perceptually similar while still containing inaccurate structures or details.

Perceptual objectives can also introduce artifacts when used for training, although this does not mean LPIPS itself should automatically be treated as a training loss.

### Project Context

KLA confirms LPIPS as one of the evaluation metrics for the benchmark.

For this project, LPIPS should therefore be treated as an **evaluation metric**, not automatically as a training objective.

### Evidence Classification

**Direct Literature Evidence** for the metric.

**KLA/Project-Confirmed** for its role in benchmark evaluation.

---

## 4. Metric Comparison

| Property | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Primary information | Pixel fidelity | Structural similarity | Learned perceptual similarity |
| Better direction | Higher | Higher | Lower |
| Pixel sensitivity | High | Moderate | Lower |
| Structural sensitivity | Limited | High | Moderate/High |
| Perceptual sensitivity | Low | Moderate | High |
| Noise sensitivity | High | Context-dependent | Texture-dependent |
| Geometric changes | Highly sensitive to pixel shifts | More structurally aware | Feature-space comparison |
| Major limitation | May not reflect perceptual quality | Can be insensitive to uniform biases | Perceptual similarity does not guarantee physical correctness |

This comparison should not be interpreted as proving that one metric is universally superior. Each metric measures a different aspect of restoration quality.

---

## 5. Metric Disagreement

One of the most important findings for this project is that different metrics can rank restoration methods differently.

### NAFNet vs Restormer

In SEM experiments, NAFNet achieved higher numerical PSNR and SSIM than Restormer.

However, Restormer was reported to preserve circular geometry more faithfully under distribution shift.

This demonstrates that higher pixel-wise and structural scores do not necessarily guarantee better geometric fidelity.

**Evidence Classification:** SEM/Microscopy Evidence.

### Autoencoder vs U-Net with Attention

In SEM restoration experiments, an Autoencoder achieved the highest reported PSNR, while U-Net with Attention achieved a better LPIPS/DISTS result.

This provides another example of different metrics producing different rankings between restoration approaches.

**Evidence Classification:** SEM/Microscopy Evidence.

### Pixel Loss vs Feature Loss

Research comparing pixel-based and feature-based objectives shows that models optimized for pixel-level accuracy can achieve higher PSNR, while feature-based approaches may produce sharper fine details despite lower PSNR.

This demonstrates the difference between numerical reconstruction accuracy and perceptual detail preservation.

**Evidence Classification:** Direct Literature Evidence.

---

## 6. Why Multiple Metrics Are Needed

The literature and project documentation support using multiple evaluation metrics because no individual metric captures every aspect of restoration quality.

### PSNR

Provides information about:

- Pixel-level reconstruction accuracy
- Numerical fidelity
- Noise/error reduction

### SSIM

Provides additional information about:

- Local structure
- Contrast
- Luminance
- Structural similarity

### LPIPS

Provides additional information about:

- Learned feature similarity
- Perceptual differences
- Texture and visual appearance

Together, these metrics provide a broader evaluation than any single metric.

However, using multiple metrics does not mathematically guarantee a better evaluation. Their results still need to be interpreted alongside qualitative and task-specific analysis.

### Evidence Classification

**Strongly Supported Interpretation**

---

## 7. SEM-Specific Limitations

### Circular Geometry

SEM research reports cases where models with strong quantitative scores nevertheless produced geometric distortions in circular structures.

This is particularly important for semiconductor metrology, where geometric accuracy can be more important than purely visual similarity.

### Oversmoothing

FIB-SEM research reports that some denoising models can produce excessive smoothing and loss of fine pore textures.

A model can therefore achieve measurable numerical improvement while reducing useful structural detail.

### Fine Details

Pixel-wise metrics do not necessarily capture whether fine structures remain visually or physically meaningful after restoration.

### Acquisition Variation

SEM restoration performance can change when acquisition parameters such as focus, beam alignment, or accelerating voltage vary.

Therefore, a high score on one evaluation distribution does not automatically establish robustness to acquisition variation.

### Hallucinated or Artificial Details

Perceptual restoration approaches can produce visually convincing details that are not present in the reference.

Consequently, perceptual similarity should not be interpreted as proof of physical correctness.

### Evidence Classification

**SEM/Microscopy Evidence**

---

## 8. KLA / Project Context

KLA confirms that the benchmark considers:

- PSNR
- SSIM
- LPIPS

The exact weighting used for the final ranking is not publicly disclosed.

Therefore, this research does not assume:

- A specific KLA weighting formula
- That PSNR is the sole optimization target
- That SSIM is more important than LPIPS
- That LPIPS is the primary metric
- That any individual metric is optimal for the benchmark

The metrics should be treated independently unless an official project source establishes a specific relationship.

### Evidence Classification

**KLA/Project-Confirmed Fact**

---

## 9. Research Gaps

The following questions remain unresolved for this specific SEM benchmark.

### KLA Metric Weighting

The exact relative weighting of PSNR, SSIM, and LPIPS in the final evaluation is unknown.

Therefore, the mathematically optimal trade-off between these metrics cannot be determined from the available information.

### LPIPS on Grayscale SEM

The provided literature does not establish how reliably LPIPS represents expert judgment for the specific grayscale SEM structures in this benchmark.

### Expert Judgment

There is insufficient evidence to establish that any single metric consistently correlates best with semiconductor-specific expert judgment.

### Degradation Severity

The behavior of PSNR, SSIM, and LPIPS under different levels of the benchmark's multiplicative speckle noise has not been fully established.

### Distribution Shift

It remains unknown how metric rankings behave when the hidden evaluation data differs in acquisition conditions or degradation severity from the available training data.

### Evidence Classification

**Unknown / Project Inference**

---

## 10. Evidence Summary

| Finding | Evidence Classification |
|---|---|
| PSNR measures pixel-wise reconstruction fidelity | Direct Literature Evidence |
| SSIM measures luminance, contrast, and structure | Direct Literature Evidence |
| LPIPS measures learned perceptual feature similarity | Direct Literature Evidence |
| PSNR may disagree with perceptual or structural quality | Direct Literature Evidence |
| SEM models can show geometric distortion despite strong numerical metrics | SEM/Microscopy Evidence |
| Restormer preserved circular geometry better under distribution shift despite lower PSNR | SEM/Microscopy Evidence |
| Different metrics can rank restoration methods differently | Direct + SEM/Microscopy Evidence |
| KLA evaluates PSNR, SSIM, and LPIPS | KLA/Project-Confirmed |
| Exact KLA metric weighting is unknown | KLA/Project-Confirmed / Unknown |
| No single metric is established as optimal for this benchmark | Project Inference |
| LPIPS behavior for grayscale SEM structures requires further validation | Unknown |

---

## Conclusion

The literature establishes that PSNR, SSIM, and LPIPS measure different aspects of restoration quality.

PSNR provides strong pixel-level fidelity information but does not fully represent perceptual or geometric quality. SSIM provides additional structural information but has limitations related to luminance, contrast, and scale. LPIPS provides learned perceptual similarity information that can reveal differences not captured by pixel-wise metrics.

SEM-specific evidence demonstrates that these metrics can disagree with structural or visual inspection. In particular, a model with higher PSNR and SSIM may not necessarily preserve geometric structures as faithfully as another model.

Therefore, the use of PSNR, SSIM, and LPIPS together provides a more comprehensive evaluation framework than relying on a single metric.

The exact KLA metric weighting and the behavior of these metrics under the benchmark's specific degradation distribution remain unknown and require project-specific validation.