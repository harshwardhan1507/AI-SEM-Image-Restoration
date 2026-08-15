# Qualitative Restoration Evaluation and Failure Modes

## Objective

Review research concerning qualitative evaluation of image restoration and common restoration failure modes, with particular attention to SEM and microscopy imagery.

The project evaluates restored SEM images using numerical metrics, but KLA also recommends inspecting actual restored images rather than relying exclusively on numerical scores.

This research provides context for the project's qualitative evaluation workflow in Issue #42.

---

## 1. Why Numerical Metrics Are Not Sufficient

Numerical image-restoration metrics such as PSNR and SSIM provide useful quantitative measurements, but they do not fully represent visual or structural restoration quality.

Pixel-level metrics aggregate errors across an image. As a result, localized problems affecting a small but important structure may have limited influence on the final numerical score.

The reviewed literature demonstrates that restoration methods can obtain strong quantitative results while still exhibiting visually important problems such as:

- Oversmoothing
- Fine-detail loss
- Geometric distortion
- Residual noise
- Edge degradation
- Artificial textures
- Hallucinated structures

This does not mean that PSNR or SSIM are unreliable metrics. Rather, they measure specific aspects of restoration quality and should be interpreted alongside qualitative evidence.

### Evidence Classification

**Direct Literature Evidence**

---

## 2. Qualitative Evaluation

### Why Visual Inspection Is Useful

Visual inspection provides information about localized structures and artifacts that may not be apparent from aggregate numerical metrics.

This is particularly important for SEM imagery because small structures, boundaries, textures, and geometric features can carry important scientific or metrology information.

The reviewed literature contains examples where visual or expert assessment differs from numerical metric rankings.

KLA also explicitly recommends inspecting restored images rather than relying only on numerical metrics.

### Evidence Classification

**SEM/Microscopy Evidence**

**KLA/Project-Confirmed**

---

## 3. Restoration Failure Modes

### Oversmoothing and Fine-Detail Loss

Oversmoothing occurs when a restoration removes not only unwanted noise but also useful high-frequency structures and fine textures.

Xu et al. report that denoising can improve quantitative performance while introducing noticeable smoothing and blurring and reducing fine texture.

For microscopy images, this can affect the visibility of small structural boundaries and pore features.

### Evidence Classification

**Direct SEM/Microscopy Evidence**

---

### Geometric Distortion

Restoration models can alter the geometry of structures even when their overall numerical scores remain strong.

Park et al. report pronounced geometric distortions in circular SEM structures, demonstrating that numerical restoration quality does not necessarily guarantee preservation of geometric fidelity.

This is particularly relevant to SEM applications involving structured features such as circular hole arrays.

### Evidence Classification

**Direct SEM Evidence**

---

### Residual Noise and Edge Artifacts

A restoration model may remove much of the unwanted degradation while leaving residual high-frequency noise or halo-like artifacts.

The reviewed literature also reports cases of softened or degraded edges.

These effects can be difficult to characterize using only an aggregate image-level score because the problem may be concentrated around specific structures or boundaries.

### Evidence Classification

**Direct / Qualified Literature Evidence**

The specific evidence comes primarily from the reviewed restoration literature and should not automatically be treated as experimentally established behavior on this SEM benchmark.

---

### Splotchy Artifacts

The reviewed restoration literature reports visible splotchy artifacts in relatively flat or texture-less regions under certain L2/MSE-based restoration configurations.

This demonstrates that a restoration objective can produce visually noticeable artifacts even when the overall numerical reconstruction error is low.

### Evidence Classification

**Qualified General Restoration Evidence**

---

### Artificial and Periodic Textures

The reviewed literature reports artificial periodic patterns, including cross-hatch and grating-like artifacts, under particular restoration and super-resolution configurations.

These findings demonstrate that restoration models can introduce structured artifacts that are visually distinguishable from the expected image content.

However, these artifacts were not all demonstrated specifically on the project's SEM benchmark.

### Evidence Classification

**Direct General Restoration Evidence**

---

### Hallucinated Structures

Generative restoration approaches can introduce structures that were not present in the original observation.

Shin et al. report this type of failure in microscopy/SEM-related restoration research.

This is particularly important for scientific imaging because a visually convincing structure is not necessarily a physically correct structure.

### Evidence Classification

**Direct SEM/Microscopy Evidence**

---

## 4. Qualitative Evaluation Methods

The reviewed literature uses several approaches to evaluate restoration results visually and structurally.

### Side-by-Side Visual Comparison

Restoration studies commonly compare:

- Input image
- Ground-truth/reference image
- Restored image
- Other restoration outputs

This allows differences in structure, detail, artifacts, and noise removal to be inspected directly.

### Crop and Zoom Inspection

Zoomed or cropped regions are used to inspect details that may not be visible at the full-image scale.

Typical targets include:

- Fine structures
- Edges
- Textures
- Small artifacts
- Localized detail loss

### Residual / Error Maps

Some studies visualize the absolute difference between prediction and ground truth:

`|GT - Pred|`

Residual maps provide spatial information about where restoration errors remain.

They should be treated as a complementary diagnostic method rather than a universally superior evaluation technique.

### Expert / Human Evaluation

The reviewed literature includes expert or human evaluation approaches for assessing perceptual and structural quality.

These evaluations can reveal cases where human or expert preferences disagree with numerical metric rankings.

### Downstream Structural Validation

Some microscopy studies evaluate restoration by examining its effect on downstream measurements such as:

- Porosity
- Pore-size statistics
- Other structural measurements

This provides task-specific information about whether restoration preserves information needed for subsequent analysis.

### Histogram Analysis

Grayscale histogram comparisons can be used to examine intensity distributions, contrast differences, and other distributional changes after restoration.

### Evidence Classification

**Direct Literature Evidence**

---

## 5. Metric and Visual Disagreement

One of the most important findings for this project is that numerical metric rankings do not necessarily correspond perfectly to visual or expert assessment.

### Restormer vs NAFNet

In SEM experiments reported by Park et al., NAFNet achieved higher PSNR and SSIM, while Restormer was reported to preserve circular geometry more faithfully under distribution shift.

This demonstrates that stronger numerical scores do not necessarily guarantee better geometric preservation.

### Autoencoder vs U-Net with Attention

In the reviewed SEM restoration work, an Autoencoder achieved the highest reported PSNR/SSIM, while U-Net with Attention achieved stronger perceptual evaluation and was preferred by experts for structural realism.

This provides another example of different evaluation criteria producing different rankings.

### Evidence Classification

**Direct SEM/Microscopy Evidence**

---

## 6. Relationship Between Quantitative and Qualitative Evaluation

The literature supports treating quantitative and qualitative evaluation as complementary.

### Quantitative Metrics

Metrics such as PSNR, SSIM, and LPIPS provide reproducible numerical information about restoration quality.

They are useful for:

- Comparing experiments
- Tracking improvement
- Ranking model outputs according to specific metric definitions
- Reporting reproducible results

### Qualitative Evaluation

Visual inspection provides additional information about:

- Structural fidelity
- Fine-detail preservation
- Geometric accuracy
- Residual noise
- Edge quality
- Artificial textures
- Hallucinated structures
- Localized artifacts

Therefore, numerical improvements should be interpreted together with visual evidence rather than treated as proof of complete restoration quality.

### Evidence Classification

**Strongly Supported Interpretation**

---

## 7. SEM-Specific Relevance

SEM restoration has characteristics that make qualitative inspection particularly important.

### Geometric Structures

Circular structures and other small geometric features can be distorted during restoration.

Such changes may be important even when their contribution to the overall numerical error is relatively small.

### Fine Structures

SEM images may contain small boundaries, pores, lines, and other high-frequency structures that can be reduced by excessive smoothing.

### Structural Fidelity

A restoration should not only remove unwanted degradation but also preserve meaningful structures present in the original image.

### Artificial Structures

Hallucinated or artificial structures are particularly concerning in scientific imaging because visually plausible content does not necessarily represent physically observed content.

### Evidence Classification

**SEM/Microscopy Evidence + Project Relevance**

---

## 8. KLA / Project Context

KLA recommends that teams inspect actual restored images rather than relying exclusively on numerical metrics.

The parent research issue also identifies qualitative evaluation as an explicit research area and links this research to Issue #42.

Therefore, qualitative inspection is part of the project's broader evaluation methodology rather than a replacement for PSNR, SSIM, or LPIPS.

The exact implementation of Issue #42 remains a project-specific decision.

### Project Diagnostic Region

The `[64:192, 64:192]` crop used by the project is a project-specific diagnostic region.

It should not be described as a standard coordinate established by SEM literature.

### Tiling Seams

Tiling or sliding-window inference can introduce discontinuities at tile boundaries.

This provides a reasonable deployment risk to inspect, but the available evidence does not establish that tiling seams necessarily occur in this project's NAFNet implementation.

### Evidence Classification

**KLA/Project-Confirmed** for the recommendation to inspect restored images.

**Project-Specific Decision** for the diagnostic crop.

**Qualified Project Inference** for tiling-seam inspection.

---

## 9. Research Gaps

The following questions remain unresolved for the specific SEM benchmark.

### Oversmoothing Threshold

The literature does not establish a specific threshold at which oversmoothing begins to compromise the physical accuracy required by this project's metrology objectives.

This requires project-specific validation.

### Multiplicative Speckle Sensitivity

The exact sensitivity of perceptual metrics such as LPIPS or DISTS to the benchmark's specific multiplicative speckle component has not been established.

Results from other degradation types should not automatically be assumed to apply to this benchmark.

### Benchmark-Specific Failure Modes

Although the literature documents several restoration failure modes, it does not establish that every documented failure will occur on this project's SEM benchmark.

The occurrence and severity of each failure mode must be determined through project evaluation.

### Metric Weighting

KLA confirms PSNR, SSIM, and LPIPS as evaluation metrics, but the exact weighting used for final evaluation is not publicly disclosed.

Therefore, no specific weighted KLA score should be assumed.

### Evidence Classification

**Unknown / Project Inference**

---

## 10. Evidence Classification

The findings in this issue are separated into four categories.

### Direct Literature Evidence

The source explicitly demonstrates or reports the finding.

Examples:

- Oversmoothing in microscopy restoration
- Geometric distortion in SEM structures
- Hallucinated structures in the cited microscopy research

### Strongly Supported Interpretation

Multiple findings support a broader interpretation.

Example:

- Quantitative and qualitative evaluation should be treated as complementary.

### Project-Specific Inference

The literature provides a reasonable basis for considering something during project evaluation, but it has not been experimentally established on this benchmark.

Examples:

- Tiling-seam inspection
- Relevance of specific diagnostic regions

### Unknown / Insufficient Evidence

The available sources do not establish a reliable conclusion.

Examples:

- Exact acceptable oversmoothing threshold
- Exact LPIPS/DISTS sensitivity to benchmark-specific speckle
- Whether every literature-reported artifact occurs on this dataset

---

## 11. Key Takeaways

1. **Numerical metrics are incomplete:** PSNR, SSIM, and LPIPS provide useful quantitative information, but numerical scores alone do not fully establish visual or structural restoration quality.
2. **Oversmoothing is a documented failure mode:** Microscopy research demonstrates that denoising can improve numerical metrics while reducing fine texture and detail.
3. **Geometric fidelity matters:** SEM research demonstrates that strong numerical scores can coexist with geometric distortion.
4. **Restoration can introduce artifacts:** The literature documents residual noise, edge degradation, artificial textures, and hallucinated structures under different restoration configurations.
5. **Metric and expert rankings can disagree:** SEM-related research provides direct examples where numerical metrics and expert/structural assessments do not produce identical rankings.
6. **Qualitative and quantitative evaluation are complementary:** Visual comparisons, zoomed inspection, residual maps, expert evaluation, and downstream structural analysis provide additional information alongside numerical metrics.
7. **Literature findings must not be overgeneralized:** A failure demonstrated in general image restoration or super-resolution literature does not prove that the same failure occurs on this SEM benchmark.
8. **Issue #42 is research-supported:** KLA guidance and the reviewed literature provide a research basis for inspecting restored images alongside numerical evaluation.

---

## Conclusion

The reviewed literature establishes that qualitative inspection provides important information that cannot be fully represented by aggregate numerical restoration metrics.

SEM and microscopy research demonstrates failure modes including oversmoothing, fine-detail loss, geometric distortion, and hallucinated structures, while broader restoration literature provides evidence for additional artifacts such as splotching, periodic textures, residual noise, and edge degradation.

These findings support the project's decision to evaluate restored images visually alongside PSNR, SSIM, and LPIPS.

The specific behavior of these failure modes on the project's benchmark remains an empirical question and should not be assumed from literature alone.

**Evidence Classification:** Direct Literature Evidence + SEM/Microscopy Evidence + KLA/Project-Confirmed Context + Project-Specific Inference