# Issue #55 — Review of External Datasets and Pretrained Model Usage

## Objective

Document research and best-practice considerations for utilizing external datasets and pretrained restoration models in Scanning Electron Microscopy (SEM) image restoration, in compliance with KLA hackathon guidelines and academic literature.

## Parent Issue

[#45 — Analyze research literature and technical foundations for SEM image restoration](https://github.com/harshwardhan1507/AI-SEM-Image-Restoration/issues/45)

---

## 1. Governance: KLA Explicit Rules vs. Academic Literature Recommendations

A fundamental requirement of this issue is to clearly distinguish between **KLA-confirmed competition rules** and **academic literature recommendations**.

| Dimension | KLA Explicitly Permitted / Required Rules | Academic Literature Recommendations & Findings |
|---|---|---|
| **External Datasets** | **Permitted**: Any publicly available dataset, provided its license permits hackathon usage. Mandatory documentation required if used. | Pretraining or augmenting with external datasets (e.g., DIV2K, SIDD) improves low-level feature extraction, but domain shift must be actively managed. |
| **Pretrained Model Weights** | **Permitted**: Any publicly available pretrained restoration model weights (e.g., NAFNet on SIDD, Restormer, SwinIR), provided license permits. | Pretrained weights serve as superior initializers compared to random initialization; fine-tuning with small learning rates preserves low-level edge filters while adapting to target SEM noise. |
| **Architectures** | **Permitted**: Open-source model architectures and custom modifications. | Lightweight restoration architectures with simplified attention mechanisms (e.g., NAFNet) provide optimal trade-offs between PSNR/SSIM gains and inference latency. |
| **Domain Adaptation** | **Permitted**: Fine-tuning and domain adaptation techniques on external or synthetic data. | Synthetic degradation matching (aligning Gaussian noise, speckle noise, and 2× downsampling) is critical when utilizing external natural or microscopy data. |
| **Documentation Standards** | **Mandatory**: If external datasets or pretrained weights are used, they MUST be documented with Name, Link, License, Research Paper/Model Card. | Standardized Data Cards and Model Cards ensure reproducibility, legal compliance, and provenance tracking. |

**Evidence Classification:** KLA/Project-Confirmed Fact & Direct Evidence

---

## 2. Transfer Learning for Image Restoration & Super-Resolution

### 2.1 Low-Level Feature Transferability vs. High-Level Semantic Shift

- **Mechanisms**: Transfer learning in high-level vision tasks (e.g., classification, object detection) relies on abstract semantic representations (ImageNet features). In contrast, low-level vision tasks (denoising, super-resolution, artifact removal) rely on local primitives: edge filters, directional gradients, noise statistics, and spatial spatial-frequency textures.
- **Domain Distance**: Natural RGB images (e.g., ImageNet, DIV2K, SIDD) feature 3-channel color, natural lighting gradients, and organic shapes. SEM images are single-channel grayscale, electron-beam intensity maps dominated by Poisson/Gaussian noise, multiplicative speckle noise, edge bloom, and sub-micron semiconductor geometry (lines, vias, gratings).
- **Transfer Dynamics**: Pretraining on natural image super-resolution datasets (DIV2K, Flickr1024) teaches convolutional kernels general edge-sharpening and high-frequency spatial reconstruction. However, zero-shot inference across domain distance results in performance degradation. Fine-tuning on target SEM data adapts these low-level filters to SEM noise statistics.

### 2.2 Fine-Tuning Strategies: Layer-Wise vs. Full Model Fine-Tuning

- **Full Model Fine-Tuning**: Training all model parameters with a small initial learning rate (e.g., \(1 \times 10^{-4}\) to \(1 \times 10^{-5}\)) using cosine annealing allows the model to retain low-level edge kernels while adapting high-level attention and residual blocks to SEM artifacts.
- **Encoder-Freezing / Partial Fine-Tuning**: Freezing early feature extraction layers preserves basic edge and corner detectors. However, empirical literature in image restoration demonstrates that full fine-tuning with warm-up yields superior PSNR/SSIM compared to frozen encoder backbones, because low-level noise characteristics in SEM differ significantly from natural noise.

### 2.3 Risk Analysis: Natural Image Prior Hallucination

- **Hallucination Risk**: Highly expressive generative or deep pretrained models (e.g., GANs, diffusion models pretrained on natural scenes) risk hallucinating organic textures, smoothing out sharp semiconductor line edges, or introducing false features.
- **Mitigation**: Constrain restoration models to deterministic regression architectures (such as NAFNet) or enforce strict structural and pixel-wise fidelity metrics (L1/L2 loss, SSIM, structural frequency loss) during fine-tuning on SEM data.

**Evidence Classification:** Strongly Supported (Academic Literature Consensus)

---

## 3. Review of Pretrained Restoration Models

### 3.1 NAFNet (Nonlinear Activation Free Network)

- **Pretrained Benchmarks Available**: SIDD (Smartphone Image Denoising Dataset - natural noisy RGB), REDS (Video deburring/restoration), GoPro (Motion blur).
- **Source / Repository**: [Megvii Research NAFNet GitHub](https://github.com/megvii-research/NAFNet)
- **License**: MIT License.
- **Paper / Model Card**: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
- **Transfer Efficacy to SEM**: **High**. NAFNet's SimpleGate and Simplified Channel Attention blocks extract spatial and channel interactions without non-linear activations, avoiding thresholding artifacts on single-channel grayscale SEM signals. Pretrained SIDD weights provide strong baseline denoising kernels.

### 3.2 Restormer (Efficient Transformer for Image Restoration)

- **Pretrained Benchmarks Available**: SIDD, Motion Blur (GoPro), Raindrop, Real Denoising.
- **Source / Repository**: [Restormer GitHub](https://github.com/swz30/Restormer)
- **License**: Apache-2.0 License.
- **Paper / Model Card**: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022.
- **Transfer Efficacy to SEM**: **Medium-High**. Multi-Dconv Head Transposed Attention (MDTA) captures non-local spatial context effectively for repeated pattern SEM grids. However, higher computational complexity requires careful tile-based inference.

### 3.3 SwinIR / HAT (Hybrid Attention Transformer)

- **Pretrained Benchmarks Available**: DIV2K (2×, 3×, 4× Classical Super-Resolution), Real-ESRGAN datasets.
- **Source / Repository**: [SwinIR GitHub](https://github.com/JingyunLiang/SwinIR), [HAT GitHub](https://github.com/XPixelGroup/HAT)
- **License**: Apache-2.0 / MIT Licenses.
- **Paper / Model Card**: Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCV 2021; Chen et al., "Activating More Pixels in Image Restoration Transformer", CVPR 2023.
- **Transfer Efficacy to SEM**: **High** for 2× Spatial Reconstruction. SwinIR's shifted-window self-attention effectively handles 2× upscaling of degraded inputs (128×128 to 256×256).

### 3.4 EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution)

- **Pretrained Benchmarks Available**: DIV2K, DF2K (DIV2K + Flickr2K).
- **Source / Repository**: [EDSR-PyTorch GitHub](https://github.com/sanghyun-son/EDSR-PyTorch)
- **License**: MIT License.
- **Paper / Model Card**: Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution", CVPR Workshops 2017.
- **Transfer Efficacy to SEM**: **High** baseline stability. Standard CNN residual architecture without attention modules; reliable, low-overhead initialization.

**Evidence Classification:** Direct Evidence & KLA/Project-Confirmed Fact

---

## 4. Domain Adaptation & Synthetic Degradation Alignment

### 4.1 Bridging Domain Shift via Degradation Matching

When leveraging external datasets or natural image pretrained weights, the primary cause of poor performance on SEM evaluation data is **domain shift in noise distribution**.

- **Confirmed KLA Degradations**:
  1. Additive Gaussian Noise
  2. Multiplicative Speckle Noise
  3. 2× Downsampling (Spatial Resolution Recovery)
  4. Non-fixed degradation sequence.
- **Domain Adaptation Protocol**:
  If external clean HR datasets (e.g., DIV2K clean images) are used for pretraining or data augmentation, they MUST be subjected to synthetic degradation pipelines that accurately simulate the exact KLA degradation order and intensity ranges (Gaussian variance \(\sigma^2\), speckle factor \(\gamma\), downsampling kernels).

### 4.2 Supervised Domain Adaptation (Two-Stage Fine-Tuning)

- **Two-Stage Training Protocol**:
  1. **Stage 1 (Pretraining / Warm-up)**: Pretrain model backbone on synthetic degraded pairs from external datasets (or load pretrained weights on SIDD / DIV2K).
  2. **Stage 2 (Target Adaptation)**: Fine-tune the network exclusively on the target paired SEM dataset (128×128 degraded inputs \(\to\) 256×256 clean targets) with a reduced learning rate.
- **Benefits**: Combines generic structural super-resolution priors with specialized SEM noise removal and semiconductor edge sharpening.

**Evidence Classification:** Strongly Supported (Academic & Experimental Best Practice)

---

## 5. Survey of External Datasets & Licensing Audit

The table below summarizes candidate external datasets evaluated for SEM restoration pretraining or data augmentation, including their official license classifications and hackathon permissibility.

| Dataset Name | Domain / Task | Source / Link | License | Hackathon Permissibility | Notes & Transfer Suitability |
|---|---|---|---|---|---|
| **DIV2K** | Natural HR Super-Resolution | [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | Custom Open Academic / Permissive | **Permitted** | 800 HR training images; standard benchmark for 2× super-resolution pretraining. |
| **Flickr1024** | Natural / Stereo Super-Resolution | [Flickr1024 GitHub](https://github.com/yingqianwang/Flickr1024) | Permissive Academic | **Permitted** | High-quality 1024×1024 images; suitable for low-level edge pretraining. |
| **Urban100** | Urban Architecture / Structural SR | [Urban100 Dataset](https://github.com/jbhuang0604/SelfExSR) | Public Domain / Academic | **Permitted** | Rich in geometric lines, grids, and repetitive structures resembling semiconductor layouts. |
| **SIDD (Smartphone Image Denoising)** | Real Noisy-Clean RGB Denoising | [SIDD Benchmark](https://www.eecs.yorku.ca/~kamel/sidd/) | CC BY-NC-SA 4.0 | **CAUTION / Audit Required** | Non-commercial restriction (NC). Hackathon context requires auditing team/organizer license terms. |
| **BSD500 (Berkeley Segmentation)** | Natural Images | [BSD500 Dataset](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) | BSD License | **Permitted** | Benchmark for edge detection and classical denoising/super-resolution. |
| **FIB-SEM Digital Rock / Materials** | Materials SEM / FIB-SEM | [NIST / Materials Data](https://www.nist.gov/) | Public Domain / CC BY 4.0 | **Permitted** | Microscopy SEM data; highly relevant domain, though material contrast differs from IC chips. |
| **BioImage / CARE Benchmark** | Fluorescence & Light Microscopy | [CARE Project](https://csbdeep.bioimagecomputing.com/) | CC BY 4.0 | **Permitted** | Scientific microscopy restoration pairs; strong domain alignment with low-SNR restoration. |

> [!WARNING]
> **License Compliance Requirement:**
> Always verify the exact license terms of external datasets before downloading or utilizing them in final submissions. Datasets with **CC BY-NC** (Non-Commercial) or restricted corporate licenses must be audited against hackathon competition terms.

**Evidence Classification:** Direct Evidence (License & Source Verification)

---

## 6. Standardized Governance & Documentation Protocols

As explicitly required by KLA, any submission using external datasets or pretrained weights MUST maintain a strict documentation log. Below are the standard schema templates established for this project.

### 6.1 Dataset Documentation Schema

```markdown
### External Dataset Audit Entry
- **Dataset Name**: [e.g., DIV2K - Diverse 2K resolution high quality images]
- **Source / Repository**: [e.g., ETH Zurich Computer Vision Lab]
- **URL Link**: [e.g., https://data.vision.ee.ethz.ch/cvl/DIV2K/]
- **License Type**: [e.g., Open Academic License / CC BY 4.0]
- **Associated Research Paper**: [Agustsson et al., "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study", CVPRW 2017]
- **Usage Purpose**: [e.g., Pretraining 2× upscaling feature extraction layers before fine-tuning on SEM data]
```

### 6.2 Pretrained Model & Weights Documentation Schema

```markdown
### Pretrained Weight Audit Entry
- **Model Architecture Name**: [e.g., NAFNet-width64]
- **Pretrained Weights Checkpoint**: [e.g., NAFNet-SIDD-width64.pth]
- **Source / Repository**: [e.g., https://github.com/megvii-research/NAFNet]
- **URL Link to Weights**: [e.g., https://github.com/megvii-research/NAFNet/releases/download/v1.0.0/NAFNet-SIDD-width64.pth]
- **License Type**: [e.g., MIT License]
- **Model Card / Research Paper**: [Chen et al., "Simple Baselines for Image Restoration", ECCV 2022]
- **Adaptation Strategy**: [Full model fine-tuning on target SEM paired dataset using AdamW optimizer with cosine learning rate schedule]
```

**Evidence Classification:** KLA/Project-Confirmed Requirement

---

## 7. Summary & Key Recommendations for the Project

1. **Leverage Pretrained Weights as Initializers**: Pretrained weights from restoration models (e.g., NAFNet trained on SIDD or SwinIR on DIV2K) provide faster convergence and better initial feature representations than random initialization.
2. **Mandatory Target Fine-Tuning**: Zero-shot application of natural image models is insufficient due to severe domain shift. Models initialized with external weights must undergo fine-tuning on the 128×128 to 256×256 paired SEM dataset.
3. **Synthetic Degradation Alignment**: If using clean external HR datasets (e.g., DIV2K or Urban100) to expand training data, apply synthetic Gaussian noise, speckle noise, and 2× downsampling matching KLA's degradation profile.
4. **Strict Architectural Determinism**: Avoid unconstrained generative models that risk hallucinating false semiconductor structures. Maintain deterministic regression architectures (NAFNet, Restormer, EDSR) evaluated with pixel, structural, and frequency losses.
5. **License Verification & Documentation Compliance**: Prior to training or finalizing submission artifacts, complete the Dataset Audit and Pretrained Weight Audit forms to ensure full compliance with KLA's open-source licensing rules.

---

## Evidence Classification Index

| Classification | Meaning | Key Items in This Document |
|---|---|---|
| **KLA/Project-Confirmed Fact** | Explicit rule or mandate confirmed by KLA webinar / project guidelines | KLA permissions, mandatory documentation fields, hackathon license rules. |
| **Direct Evidence** | Verified fact directly from paper, dataset license, or repository | License types (MIT, Apache 2.0, CC BY), NAFNet/Restormer model release specs. |
| **Strongly Supported** | Consolidated consensus in academic restoration & domain adaptation literature | Low-level feature transferability, domain distance risks, fine-tuning protocols. |
| **Literature Recommendation** | Best-practice guideline recommended by researchers | Synthetic degradation matching, Urban100 pattern similarity. |
