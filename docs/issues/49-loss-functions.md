# Issue #49 — Loss Functions for Image Restoration and Super-Resolution

## Objective

Review research literature concerning losses used for image restoration and super-resolution.

The review covers:

- L1 loss
- L2 / MSE loss
- Charbonnier loss
- SSIM / MS-SSIM losses
- Perceptual losses
- LPIPS
- Frequency-domain losses
- Composite losses

The purpose is to provide evidence-backed research context for Issue #39, where loss-function alternatives may be experimentally evaluated.

---

## Project Context

The current project baseline uses:

- CharbonnierLoss
- NAFNet-based restoration
- 2x super-resolution

KLA has recommended exploring task-specific losses rather than assuming that a default loss is optimal.

KLA has also identified:

- PSNR
- SSIM
- LPIPS

as relevant evaluation metrics.

The exact weighting used for the final KLA ranking is not disclosed in the reviewed project sources.

Loss-function literature therefore provides guidance about possible trade-offs, but it does not establish which loss is optimal for this project.

---

# 1. L1 Loss

## Mathematical Objective

L1 loss measures the mean absolute difference between the predicted image and the target image.

In simple terms:

`L1 = average absolute pixel error`

## Main Properties

L1 does not penalize large errors as strongly as L2/MSE.

Research by Zhao et al. found that L1-trained restoration models can reach better minima for image-restoration objectives than models trained directly with L2.

## Relationship to PSNR

PSNR is based on MSE, so L1 does not directly optimize PSNR.

However, the reviewed literature reports cases where models trained using L1 achieved better L2-based results than models trained directly with L2.

This demonstrates that the relationship between the training objective and final PSNR is not always straightforward.

## Effect on Image Quality

The reviewed literature reports that L1 can reduce the splotchy artifacts observed with L2-based restoration.

It can therefore produce visually preferable results in some restoration settings.

## SEM Evidence

L1 has been used in published image-restoration and SEM-denoising approaches.

However, the reviewed literature does not establish that L1 is superior to Charbonnier or other losses for this project's SEM benchmark.

**Evidence Classification:** Direct Evidence for reported restoration results; Project Inference when applied to this benchmark.

---

# 2. L2 / MSE Loss

## Mathematical Objective

L2/MSE measures the mean squared pixel error between the prediction and the target.

In simple terms:

`MSE = average squared pixel error`

## Main Properties

MSE is differentiable and convex with respect to the predicted pixel values.

Under an independent and identically distributed Gaussian-noise assumption, minimizing MSE corresponds to the maximum-likelihood estimation objective.

## Relationship to PSNR

PSNR is calculated directly from MSE.

Therefore, when the same target and evaluation conditions are used:

- minimizing MSE corresponds to maximizing PSNR;
- lower MSE produces higher PSNR.

This makes MSE directly aligned with PSNR.

## Optimization Behavior

The reviewed restoration literature reports that L2 optimization can produce less desirable solutions than L1 in some image-restoration settings.

Zhao et al. report experiments where an L1-trained model achieved lower L2 error than a model trained directly with L2.

This should not be interpreted as meaning that MSE is mathematically non-convex. The observation concerns the optimization behavior of neural-network training rather than the convexity of MSE with respect to its predictions.

## Effect on Image Quality

The reviewed literature reports that L2 can produce splotchy artifacts in flat regions.

Importantly, the same research notes that L2 can preserve edge sharpness well because blurring a sharp transition can produce a large pixel-wise error.

Therefore, the common statement that L2 simply "blurs all details" is too broad.

**Evidence Classification:** Direct Evidence.

---

# 3. Charbonnier Loss

## Mathematical Objective

Charbonnier loss is a differentiable approximation of L1.

It replaces the absolute-error operation with a smooth square-root formulation containing a small epsilon value.

In simplified terms:

`Charbonnier = sqrt(error² + epsilon²)`

The project's current implementation uses an epsilon value of approximately 0.001.

## Main Properties

The reviewed literature describes Charbonnier as a differentiable variant of L1.

The smoothing term provides a stable formulation around zero error.

Some reviewed sources also report numerical and optimization benefits associated with the formulation.

## Project Context

CharbonnierLoss is the current project baseline.

KLA identifies Charbonnier as a valid baseline for the restoration task.

However, the reviewed evidence does not establish that Charbonnier is the optimal loss for this SEM benchmark.

## Relationship to Other Losses

The literature provides evidence that Charbonnier is a practical restoration loss with L1-like behavior.

It does not provide sufficient evidence to conclude that Charbonnier universally outperforms L1 or L2.

A comparison between Charbonnier and alternatives such as PSNRLoss remains a project-specific experimental question.

**Evidence Classification:**

- Direct Evidence for the general restoration properties of Charbonnier.
- Project Inference when considering its suitability for this specific benchmark.

---

# 4. SSIM and MS-SSIM Losses

## Mathematical Objective

SSIM-based losses optimize structural similarity between the prediction and target.

A common formulation is:

`SSIM loss = 1 - SSIM`

MS-SSIM extends this comparison across multiple image scales.

## Main Properties

SSIM is designed around the observation that image quality is not determined only by independent pixel differences.

It considers properties such as:

- luminance
- contrast
- local structure

This makes it complementary to purely pixel-wise objectives.

## Effect on Structural Quality

The reviewed literature reports that MS-SSIM can reduce some artifacts associated with pixel-wise losses and can provide improved structural comparisons across multiple scales.

## Limitations

The reviewed sources note that SSIM-based measures can be relatively insensitive to uniform biases.

Consequently, images can obtain good structural similarity despite differences in brightness or other global characteristics.

## Project Relevance

SSIM-based losses could potentially provide an additional structural objective, but the literature does not establish that they are necessary or optimal for this SEM benchmark.

**Evidence Classification:** Direct Evidence for the properties of SSIM/MS-SSIM; Project Inference for application to this benchmark.

---

# 5. Perceptual Losses

## Mathematical Objective

Perceptual losses compare feature representations rather than directly comparing individual pixels.

A common approach extracts features from a pretrained network and measures the distance between the predicted and target feature representations.

## Effect on Detail

The reviewed literature reports that perceptual losses can reconstruct high-frequency details and visually sharp structures that pixel-wise objectives may not reproduce as strongly.

Examples in the literature include improved reconstruction of fine visual details such as small edges and textures.

## Relationship to PSNR and SSIM

Perceptual optimization can produce lower PSNR or SSIM because the objective prioritizes feature-level similarity rather than exact pixel alignment.

Therefore:

- higher perceptual similarity does not necessarily mean higher PSNR;
- higher PSNR does not necessarily mean better perceptual quality.

## Limitations

The reviewed sources report that perceptual losses can introduce artifacts, including cross-hatch patterns in some restoration settings.

They can therefore introduce a trade-off between perceptual appearance and numerical fidelity.

**Evidence Classification:** Direct Evidence for the reported natural-image restoration results.

---

# 6. LPIPS

## What LPIPS Measures

LPIPS is a learned perceptual image similarity metric based on deep feature representations.

Lower LPIPS values indicate greater perceptual similarity according to the metric.

## Project Status

KLA identifies LPIPS as an evaluation metric.

Importantly, the project documentation explicitly distinguishes LPIPS as an evaluation metric rather than assuming it as a training loss.

Therefore:

`LPIPS evaluation metric != LPIPS training objective`

## SEM Evidence

LPIPS and related perceptual metrics such as DISTS are used in SEM restoration research to provide information beyond PSNR and SSIM.

These metrics can provide complementary information about perceptual and structural restoration quality.

## Unknown

The reviewed sources do not establish whether directly optimizing LPIPS would improve the final KLA score for this grayscale SEM benchmark.

**Evidence Classification:**

- Direct Evidence as an evaluation metric.
- Unknown / Insufficient Evidence as a training loss for this project.

---

# 7. Frequency-Domain Losses

## Main Objective

Frequency-domain losses operate on representations of image frequency content rather than relying entirely on spatial-domain pixel differences.

The reviewed literature motivates these losses partly through the concept of spectral bias: neural networks can learn lower-frequency image components more readily than high-frequency components.

## Effect on Detail

Frequency-aware objectives can explicitly encourage reconstruction of high-frequency information.

This can be useful when fine structures and edges are important.

## Guided Frequency Loss

The reviewed Guided Frequency Loss work combines frequency-aware components with spatial restoration objectives.

It reports improvements across several restoration experiments and investigates cases where frequency information is particularly important.

## SEM Relevance

KLA has identified frequency-domain training as a valid research direction.

The reviewed natural-image literature also reports increased utility on constrained or repetitive data.

However, applying this result directly to semiconductor SEM hole-arrays is an inference.

The sources do not establish that frequency-domain losses are superior for this project's SEM dataset.

**Evidence Classification:**

- Direct Evidence for the reported natural-image experiments.
- Strongly Supported / Project Inference for possible relevance to structured SEM imagery.

---

# 8. Composite Losses

Composite losses combine multiple objectives in order to balance different aspects of restoration.

Examples found in the reviewed literature include:

- L1 + MS-SSIM
- MSE + SSIM
- Charbonnier + frequency-domain components

## L1 + MS-SSIM

Zhao et al. reported that their combined L1 and MS-SSIM objective performed particularly well across the evaluated metrics for the specific natural-image tasks studied.

This does not establish that the same combination is optimal for SEM restoration.

## MSE + SSIM

FIB-SEM research has used a combined MSE and SSIM objective to balance pixel-level accuracy and structural similarity.

This provides SEM-specific evidence that composite objectives can be used in microscopy restoration.

## Guided Frequency Loss

The reviewed GFL approach combines Charbonnier-based spatial reconstruction with frequency-aware components.

This demonstrates one way of combining pixel-domain and frequency-domain objectives.

## Limitation

Composite losses introduce additional weighting decisions.

The relative weights determine the balance between:

- pixel fidelity
- structural similarity
- perceptual quality
- frequency/detail preservation

The literature does not establish universal weights for these objectives.

**Evidence Classification:** Direct Evidence for the specific reported experiments.

---

# 9. Loss Comparison

| Loss | Primary Objective | Main Strength | Important Limitation |
|---|---|---|---|
| L2 / MSE | Pixel accuracy | Directly aligned with PSNR | Can produce splotchy artifacts |
| L1 | Pixel accuracy | Less sensitive to large outliers; can reduce splotchy artifacts | Does not directly optimize PSNR |
| Charbonnier | Robust pixel accuracy | Smooth L1-like objective | Superiority over other losses is not established for this project |
| SSIM | Structural similarity | Explicit structural objective | Can be insensitive to uniform brightness biases |
| MS-SSIM | Multi-scale structural similarity | Captures structure across multiple scales | Does not directly optimize pixel accuracy |
| Perceptual | Feature-level similarity | Can improve visual detail and texture | May reduce PSNR/SSIM and introduce artifacts |
| LPIPS | Perceptual evaluation | Provides perceptual similarity measurement | Training-loss effectiveness for this benchmark is unknown |
| Frequency | Frequency/detail preservation | Explicitly targets high-frequency information | Effectiveness for SEM remains unverified |
| Composite | Multiple objectives | Can balance different quality criteria | Requires weighting between objectives |

---

# 10. Relationship Between Losses and Evaluation Metrics

The reviewed literature demonstrates that the training objective and evaluation metric do not necessarily need to be identical.

### MSE and PSNR

MSE and PSNR are mathematically linked.

Therefore, MSE directly targets the quantity from which PSNR is calculated.

### L1 / Charbonnier

These losses do not directly optimize PSNR but can nevertheless produce strong PSNR results in restoration experiments.

### SSIM

SSIM-based objectives explicitly target structural similarity.

### Perceptual / LPIPS

Perceptual objectives prioritize feature-level similarity rather than exact pixel correspondence.

Consequently, improvements in perceptual quality can occur without improvements in PSNR.

### Project implication

Because the benchmark considers multiple evaluation metrics, optimizing one metric alone may not guarantee the best overall benchmark result.

The exact KLA weighting remains undisclosed.

---

# 11. SEM-Specific Evidence

The reviewed literature provides several relevant observations for SEM and microscopy restoration:

1. L1 and other pixel-wise losses are used in supervised SEM restoration.
2. Charbonnier is used as a restoration objective and is the current project baseline.
3. Composite MSE + SSIM objectives have been used in FIB-SEM restoration.
4. LPIPS and DISTS have been used as complementary evaluation metrics in SEM restoration research.
5. The reviewed sources do not establish a universally optimal loss for SEM imagery.
6. The effectiveness of losses specifically against the project's combination of Gaussian noise, multiplicative speckle noise, and downsampling remains unresolved.

**Evidence Classification:** Direct Evidence for the cited SEM studies; Project Inference for application to this benchmark.

---

# 12. Project-Specific Unknowns

The following questions cannot be answered from the literature alone:

### Charbonnier vs. PSNRLoss

The project currently uses CharbonnierLoss.

Project documentation indicates that PSNRLoss directly targets PSNR, but the reviewed evidence does not establish that PSNRLoss will outperform Charbonnier on this dataset.

### Charbonnier vs. L1 / MSE

The literature provides comparisons between these losses in general restoration settings, but not a definitive comparison on this specific SEM benchmark.

### LPIPS as a Training Objective

LPIPS is confirmed as an evaluation metric.

Whether optimizing LPIPS during training improves the final KLA score is unknown.

### Frequency-Domain Losses

Frequency-aware objectives have demonstrated benefits in reported restoration experiments, but their effectiveness on this SEM dataset remains experimentally unverified.

### Composite Losses

Composite objectives have produced strong results in specific datasets and tasks.

Whether they improve the KLA benchmark score remains unknown.

### KLA Metric Weights

The exact weighting of PSNR, SSIM, and LPIPS in the final benchmark ranking is not disclosed.

Therefore, the mathematically optimal training-loss combination cannot be determined from the available information.

---

# 13. Research Context for Issue #39

The literature provides several possible directions for experimental evaluation, but none should be considered a predetermined solution.

Potential experimental comparisons include:

- Charbonnier vs. PSNRLoss
- Charbonnier vs. L1
- Charbonnier vs. MSE
- pixel-wise loss vs. structural loss
- pixel-wise loss vs. frequency-aware loss
- composite objectives

These comparisons should be evaluated using the project's existing metrics and dataset.

The literature suggests that evaluation should consider multiple dimensions rather than relying exclusively on a single metric.

However, **Issue #39 experiments must determine which objective actually performs best on the project benchmark.**

---

# 14. Evidence Classification Summary

| Classification | Meaning |
|---|---|
| Direct Evidence | Explicitly reported by the reviewed research or project documentation |
| Strongly Supported | Supported by multiple findings but requires some interpretation |
| Project Inference | Application of literature findings to the current project |
| Unknown / Insufficient Evidence | Not established by the reviewed sources |

---

# Conclusion

The reviewed literature shows that different loss functions optimize different aspects of image restoration.

- L2/MSE is directly aligned with PSNR but can produce undesirable artifacts in some restoration settings.
- L1 provides a more robust pixel-wise objective and has demonstrated strong restoration behavior.
- Charbonnier provides a smooth L1-like formulation and is a valid restoration baseline.
- SSIM/MS-SSIM explicitly emphasize structural similarity.
- Perceptual objectives prioritize feature-level and visual similarity rather than exact pixel correspondence.
- LPIPS is an important perceptual evaluation metric in the project but is not established as a training objective for this benchmark.
- Frequency-domain losses explicitly target frequency content and high-frequency reconstruction.
- Composite losses can balance multiple objectives but require appropriate weighting.

None of these findings establishes a universally optimal loss for the project's SEM restoration task.

The appropriate loss for the benchmark remains an **experimental question for Issue #39**.