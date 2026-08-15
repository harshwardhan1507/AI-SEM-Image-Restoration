# Issue #48 — NAFNet and Restoration Architecture Design

## Objective

Review research literature concerning NAFNet and related image-restoration architectures to provide evidence-backed context for architecture selection, capacity scaling, parameter efficiency, and compute-quality trade-offs.

The current project baseline is NAFNet with:

- width = 32
- enc_blocks = [1,1,1]
- mid_blocks = 1
- dec_blocks = [1,1,1]
- upscale = 2

Current project baseline:

- PSNR: 29.4118 dB
- SSIM: 0.7891
- Raw baseline PSNR: 22.9069 dB
- Improvement: +6.5049 dB

Issue #38 independently evaluates whether larger NAFNet configurations provide meaningful gains.

---

## 1. NAFNet Architecture

NAFNet (Nonlinear Activation Free Network) uses a single-stage U-Net-style architecture to reduce inter-block complexity.

Its core NAFBlock combines:

- Layer Normalization
- 1×1 convolutions
- 3×3 depthwise convolution
- SimpleGate
- Simplified Channel Attention (SCA)
- residual connections

The architecture replaces conventional nonlinear activation functions with simpler operations, particularly multiplicative gating through SimpleGate.

**Evidence Classification:** Direct Evidence.

### Relevance

These design principles provide the architectural foundation for the project's current NAFNet baseline.

---

## 2. SimpleGate and Layer Normalization

### SimpleGate

SimpleGate splits feature channels into two groups and multiplies them element-wise.

NAFNet ablations show that SimpleGate can match or slightly outperform conventional GELU activation while maintaining the simplified activation-free design.

Reported results include improvements over GELU on the evaluated restoration benchmarks.

**Evidence Classification:** Direct Evidence.

### Layer Normalization

The NAFNet ablation study found Layer Normalization highly beneficial for training stability. The paper reports that it enabled training with a learning rate approximately 10× larger and produced substantial PSNR improvements:

- +0.44 dB on SIDD
- +3.39 dB on GoPro

**Evidence Classification:** Direct Evidence.

These findings support the use of Layer Normalization within the NAFNet architecture but do not prove that identical gains will occur on the project's SEM dataset.

---

## 3. Depth and Capacity Scaling

The NAFNet paper evaluates different network depths.

In the cited GoPro experiment:

| Depth | PSNR | Latency |
|---:|---:|---:|
| 36 blocks | 32.85 dB | 177.1 ms |
| 72 blocks | 32.88 dB | 230.1 ms |

Increasing depth from 36 to 72 blocks produced only approximately +0.03 dB PSNR while increasing latency by approximately 30%.

### Interpretation

This provides direct evidence of diminishing returns from continued depth scaling in the evaluated configuration.

It does **not** establish 36 blocks as optimal for the project's SEM benchmark.

**Evidence Classification:** Direct Evidence for the published experiment; Project Inference when applied to Issue #38.

### Project relevance

Issue #38 should experimentally determine whether increasing NAFNet capacity provides meaningful gains on the actual SEM dataset.

---

## 4. Parameter and Compute Scaling

NAFNet can be scaled through its channel width and network depth.

The reviewed project documentation identifies configurations including:

| Configuration | Width | Approx. Parameters |
|---|---:|---:|
| NAFNet-Tiny | 32 | ~1M |
| Project-scale width-32 configuration | 32 | ~17M |

The NAFNet literature also evaluates models across substantially different compute scales, demonstrating that the architecture can be scaled to different computational budgets.

**Evidence Classification:** Direct Evidence.

### Important qualification

Parameter count alone does not determine restoration quality or efficiency.

Compute should be evaluated using appropriate measures such as:

- MACs/FLOPs
- parameter count
- actual inference latency
- restoration metrics

---

## 5. NAFNet Computational Trade-offs

NAFNet demonstrates strong computational efficiency on several natural-image restoration benchmarks.

For example, the NAFNet paper reports approximately 65G MACs for NAFNet compared with approximately 140G MACs for Restormer in the cited natural-image comparison.

However, the SEM comparison provides an important qualification.

For the cited 1024×768 SEM experiment:

- NAFNet: 158.92G MACs, 0.32 s latency
- Restormer: 140.99G MACs, 0.92 s latency

Therefore, NAFNet should not be described as universally lower-MAC than Restormer.

The SEM experiment instead demonstrates that NAFNet had lower measured inference latency despite its higher MAC count in that particular configuration.

**Evidence Classification:** Direct Evidence.

---

## 6. SEM Architecture Comparison

A cited SEM restoration study compares NAFNet, Restormer, HINet, and CGNet.

Validation results include:

| Model | PSNR | SSIM | MACs | Parameters |
|---|---:|---:|---:|---:|
| NAFNet | 33.37 dB | 0.8795 | 158.92G | 26.70M |
| Restormer | 29.70 dB | 0.8266 | 140.99G | 26.11M |
| HINet | 33.38 dB | 0.8796 | 170.73G | 88.67M |
| CGNet | 32.05 dB | 0.8750 | 52.11G | 119.22M |

The same study reports that:

- Restormer preserved circular geometry particularly well under distribution shift.
- HINet and CGNet exhibited pronounced geometric distortions in circular hole structures.

### Interpretation

The results demonstrate that numerical restoration quality, parameter count, MACs, latency, and geometric fidelity can produce different architecture rankings.

NAFNet therefore should not be considered automatically superior on every criterion.

**Evidence Classification:** Direct Evidence.

---

## 7. Restormer

Restormer is a Transformer-based image-restoration architecture using mechanisms including:

- Multi-Dconv Head Transposed Attention (MDTA)
- Gated-Dconv Feed-Forward Networks (GDFN)

In the cited SEM experiment, Restormer achieved lower validation PSNR than NAFNet but was reported to preserve circular geometry particularly well under distribution shift.

**Evidence Classification:** Direct Evidence.

### Relevance

This provides an important comparison point for Issue #38 and future architecture experiments, particularly if structural or geometric fidelity becomes a major concern.

The literature does not establish that Restormer is universally superior to NAFNet.

---

## 8. HINet and CGNet

The cited SEM study also evaluated HINet and CGNet.

HINet contains a two-stage restoration structure with Half Instance Normalization, while CGNet uses cascaded residual processing.

The reported parameter counts are approximately:

- HINet: 88.67M
- CGNet: 119.22M

Despite their larger parameter counts, the cited SEM study reports pronounced geometric distortions in circular hole structures.

**Evidence Classification:** Direct Evidence.

### Relevance

This demonstrates that increasing parameter count or architectural complexity does not automatically guarantee improved structural fidelity.

---

## 9. NAFSSR and Super-Resolution

NAFSSR extends NAFNet-style restoration blocks to stereo image super-resolution.

The NAFSSR paper reports state-of-the-art stereo super-resolution performance with up to 79% parameter reduction compared with competing methods.

These results were obtained on natural-image stereo datasets including Flickr1024 and KITTI.

**Evidence Classification:** Direct Evidence for the published NAFSSR experiments.

### Important limitation

The 79% parameter-reduction result is not evidence from SEM imagery.

Applying NAFSSR architectural patterns to the project's single-image SEM super-resolution task is therefore a **Project Inference**.

---

## 10. Super-Resolution Architecture Patterns

The reviewed NAFSSR architecture uses:

- a PixelShuffle-based upsampling tail;
- a global residual connection using a bilinearly upsampled input.

These are directly reported architectural components of NAFSSR.

They can provide architectural reference points for the project's 2× super-resolution component.

However, their effectiveness on the project's single-channel SEM benchmark has not been established by the cited literature.

**Evidence Classification:** Direct Evidence for NAFSSR; Project Inference for application to this project.

---

## 11. Test-Time Local Statistics Correction

The NAFNet paper identifies a train-test inconsistency that can occur when models are trained on patches but evaluated on full-resolution images.

The paper introduces Test-time Local Statistics Correction (TLC).

In the cited GoPro experiment:

- without TLC: 33.08 dB
- with TLC: 33.69 dB

This corresponds to a +0.61 dB improvement.

The NAFSSR literature uses the related TLSC terminology.

**Evidence Classification:** Direct Evidence.

### Project relevance

The finding demonstrates that differences between training patches and full-resolution inference can affect restoration quality.

Whether the same correction provides benefits on the project's SEM images remains experimentally unverified.

**Evidence Classification for SEM application:** Project Inference.

---

## 12. Current Project Baseline

The current project NAFNet baseline reports:

- PSNR: 29.4118 dB
- SSIM: 0.7891
- Raw baseline PSNR: 22.9069 dB
- Improvement: +6.5049 dB

These are project-specific experimental results and should be kept separate from published NAFNet results.

**Evidence Classification:** Direct Evidence — Project Documentation.

---

## 13. Known Limitations

The reviewed literature identifies several limitations relevant to architecture selection:

1. Restoration performance can change under acquisition and distribution shifts.
2. Increasing network depth can produce diminishing returns.
3. Parameter count alone does not guarantee improved restoration quality.
4. PSNR and SSIM may not fully capture geometric or perceptual differences.
5. Natural-image super-resolution results cannot automatically be transferred to SEM imagery.
6. Patch-based training and full-image inference can introduce train-test inconsistencies.
7. The effect of increased NAFNet capacity on this project's SEM benchmark remains experimentally unknown.
8. The effectiveness of NAFNet specifically for the project's multiplicative speckle component has not been isolated experimentally.

**Evidence Classification:** Direct Evidence where explicitly reported; Project Inference/Unknown where applied to the current benchmark.

---

## 14. Key Conclusions

### Architecture

NAFNet provides a simplified restoration architecture based on a U-Net backbone, NAFBlocks, depthwise convolutions, SimpleGate, channel processing, normalization, and residual connections.

### Capacity

Published experiments demonstrate diminishing returns from substantial increases in NAFNet depth in the evaluated configurations.

This provides useful context for Issue #38 but does not determine the optimal configuration for the SEM benchmark.

### Efficiency

NAFNet demonstrates strong quality/compute trade-offs in natural-image restoration. Its compute and latency characteristics are task- and configuration-dependent, as demonstrated by the different natural-image and SEM comparisons.

### SEM relevance

SEM-specific research provides evidence that NAFNet performs strongly on restoration tasks involving noisy SEM observations. However, other architectures such as Restormer may provide advantages in structural fidelity under distribution shift.

### Super-resolution

NAFSSR demonstrates that NAFNet-style components can be used effectively for super-resolution, but the cited results are from natural-image stereo SR. Their application to single-image SEM restoration remains a project inference.

---

## 15. Research Gaps

The following remain unresolved for the project:

- Whether increased NAFNet width or depth improves performance on the actual SEM benchmark.
- Whether additional capacity improves robustness under acquisition/distribution shifts.
- Whether NAFNet's behavior changes significantly for the benchmark's multiplicative speckle noise.
- Whether structural fidelity improves as model capacity increases.
- Whether NAFNet or an attention-based architecture provides the best quality/compute/geometry trade-off on the project dataset.
- Whether TLC/TLSC provides measurable benefits during SEM inference.

These questions require project-specific experiments and should not be answered from the literature alone.

---

## 16. Literature Context for Issue #38

The literature provides the following context for the capacity experiments:

- Depth scaling can produce diminishing returns.
- Larger models do not necessarily provide proportional restoration gains.
- Parameter count and MACs should be evaluated alongside actual inference latency.
- Structural fidelity should be monitored alongside PSNR and SSIM.
- Published scaling behavior should be treated as a reference rather than a predetermined answer for the SEM benchmark.

Issue #38 should therefore experimentally evaluate whether additional capacity produces meaningful improvements relative to its computational cost.

---

## Evidence Classification

| Classification | Meaning |
|---|---|
| Direct Evidence | Explicitly reported by published research or project documentation |
| Strongly Supported | Supported by multiple findings but requires some interpretation |
| Project Inference | Application of literature findings to this project |
| Unknown / Insufficient Evidence | Not established by the reviewed sources |