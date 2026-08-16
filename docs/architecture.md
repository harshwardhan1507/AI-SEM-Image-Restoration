# NAFNet Architecture Specification

## Overview

This repository implements **NAFNet (Nonlinear Activation Free Network)** tailored for single-channel Scanning Electron Microscope (SEM) image restoration, performing joint denoising and $2\times$ spatial super-resolution ($128 \times 128 \to 256 \times 256$).

Standard deep learning restoration models (e.g., DnCNN, UNet, Restormer, SwinIR) rely either on non-linear activation functions (ReLU, GELU, LeakyReLU) or computationally heavy self-attention mechanisms with quadratic spatial complexity $\mathcal{O}(H^2 W^2)$. NAFNet demonstrates that non-linear activation functions can be completely omitted without degrading restoration performance, replacing them with linear and multiplication operations that achieve superior computational efficiency.

---

## Architectural Diagram

![NAFNet Architecture](<../assets/diagrams/625a251c-3d16-473c-863d-e222335d086c - Copy.png>)

*Figure 1: Full NAFNet encoder-decoder architecture featuring 3-stage symmetric hierarchy, additive skip connections, PixelShuffle $2\times$ super-resolution head, and global bilinear input skip connection.*

---

## Core Architectural Components

### 1. Architectural Topology

NAFNet follows a symmetric U-Net encoder-decoder hierarchy:

- **Input Head**: A single $3 \times 3$ convolution projecting the single-channel SEM input `(B, 1, H, W)` to the base feature space `(B, C, H, W)` where $C = \text{width} = 48$.
- **Encoder Stages (Stages 0, 1, 2)**:
  - Stage 0: `NAFBlock` $\times 1$, channels $C=48$, resolution $H \times W$.
  - Downsampling 0: $2 \times 2$ strided convolution (stride 2), channels $48 \to 96$, resolution $H/2 \times W/2$.
  - Stage 1: `NAFBlock` $\times 1$, channels $C=96$, resolution $H/2 \times W/2$.
  - Downsampling 1: $2 \times 2$ strided convolution (stride 2), channels $96 \to 192$, resolution $H/4 \times W/4$.
  - Stage 2: `NAFBlock` $\times 1$, channels $C=192$, resolution $H/4 \times W/4$.
  - Downsampling 2: $2 \times 2$ strided convolution (stride 2), channels $192 \to 384$, resolution $H/8 \times W/8$.
- **Bottleneck Stage**:
  - `NAFBlock` $\times 1$, channels $C=384$, resolution $H/8 \times W/8$.
- **Decoder Stages (Stages 2', 1', 0')**:
  - Upsampling 2: $1 \times 1$ convolution ($384 \to 384$) + `PixelShuffle(2)` $\to 192$ channels, resolution $H/4 \times W/4$.
  - Lateral Skip Addition: Element-wise sum with Encoder Stage 2 output.
  - Stage 2': `NAFBlock` $\times 1$, channels $C=192$, resolution $H/4 \times W/4$.
  - Upsampling 1: $1 \times 1$ convolution ($192 \to 192$) + `PixelShuffle(2)` $\to 96$ channels, resolution $H/2 \times W/2$.
  - Lateral Skip Addition: Element-wise sum with Encoder Stage 1 output.
  - Stage 1': `NAFBlock` $\times 1$, channels $C=96$, resolution $H/2 \times W/2$.
  - Upsampling 0: $1 \times 1$ convolution ($96 \to 96$) + `PixelShuffle(2)` $\to 48$ channels, resolution $H \times W$.
  - Lateral Skip Addition: Element-wise sum with Encoder Stage 0 output.
  - Stage 0': `NAFBlock` $\times 1$, channels $C=48$, resolution $H \times W$.
- **Super-Resolution Output Tail**:
  - A $3 \times 3$ convolution projecting feature channels: $\text{Conv2d}(48, 1 \times 2^2 = 4, 3, 1, 1)$.
  - `PixelShuffle(2)` rearranging the $4$ channels into a $2\times$ spatially upscaled single-channel output `(B, 1, 2H, 2W)`.
- **Global Bilinear Residual Skip**:
  - The degraded input is bilinearly upsampled by $2\times$: `F.interpolate(x, scale_factor=2, mode='bilinear')`.
  - The upsampled input is added directly to the PixelShuffle output: $\hat{I}_{\text{restored}} = \text{PixelShuffle}(\text{Tail}(F)) + \text{Bilinear}(I_{\text{input}})$.
  - This ensures the model learns only the high-frequency residual correction (noise cancellation and sub-pixel edge synthesis).

---

## NAFBlock Internal Mechanics

![NAFBlock Mechanics](<../assets/diagrams/96faf70b-9a3a-4d39-b60f-eb7d15242125 - Copy.png>)

*Figure 2: Detailed internal structure of the Nonlinear Activation Free Block (NAFBlock).*

A `NAFBlock` consists of two sequential residual sub-branches: Spatial/Channel Mixing (Attention branch) and Feed-Forward Network (FFN branch).

### 1. Spatial Attention Branch
1. **LayerNorm2d**: Channel-wise normalization preserving 2D spatial layouts.
2. **Pointwise Expansion Conv**: $1 \times 1$ convolution expanding channels $C \to 2C$.
3. **Depthwise Conv**: $3 \times 3$ depthwise convolution (groups=$2C$, padding=1) extracting spatial context.
4. **SimpleGate**: Channel-splitting element-wise multiplication:
   $$\text{SimpleGate}(X_1, X_2) = X_1 \odot X_2 \quad (X_1, X_2 \in \mathbb{R}^{B \times C \times H \times W})$$
5. **Simplified Channel Attention (SCA)**:
   - Global Average Pooling (GAP) collapsing spatial dimensions to $(B, C, 1, 1)$.
   - $1 \times 1$ pointwise convolution computing inter-channel weights.
   - Channel scaling via element-wise multiplication: $\text{SCA}(X) = X \odot \mathcal{F}_{\text{Linear}}(\text{GAP}(X))$.
6. **Pointwise Projection Conv**: $1 \times 1$ convolution mapping channels back $C \to C$.
7. **Dropout & Learnable Scale**: Scaled by learnable parameter $\beta$ and added to the block input: $X' = X + \beta \cdot \text{Attention}(X)$.

### 2. Feed-Forward Network (FFN) Branch
1. **LayerNorm2d**: Normalization across channels.
2. **Pointwise Expansion Conv**: $1 \times 1$ convolution expanding channels $C \to 2C$ (expansion factor 2).
3. **SimpleGate**: Splitting $2C$ channels into two $C$-channel tensors and computing element-wise product $X_1 \odot X_2$.
4. **Pointwise Projection Conv**: $1 \times 1$ convolution mapping channels $C \to C$.
5. **Dropout & Learnable Scale**: Scaled by learnable parameter $\gamma$ and added to the branch input: $Y = X' + \gamma \cdot \text{FFN}(X')$.

---

## Architectural Comparison

| Architecture | Paradigm | Nonlinearity | Attention Mechanism | Spatial Complexity | Memory Footprint | SEM Micrograph Suitability |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **DnCNN** | Feed-forward CNN | ReLU | None | $\mathcal{O}(HW)$ | Low | Poor on compound Poisson-speckle noise |
| **UNet** | Encoder-Decoder | ReLU / LeakyReLU | None | $\mathcal{O}(HW)$ | Moderate | Moderate (Over-smooths fine contact edges) |
| **Restormer** | Transformer | GELU | Multi-Dconv Transposed Attention | $\mathcal{O}(H^2 W^2)$ | High | High compute cost for large wafer fields |
| **SwinIR** | Swin Transformer | GELU | Shifted Window Self-Attention | $\mathcal{O}(HW \cdot W_{\text{win}}^2)$ | High | Latency bottleneck in inline inspection |
| **NAFNet (Ours)** | Activation-Free CNN | **SimpleGate** | **Simplified Channel Attention (SCA)** | **$\mathcal{O}(HW)$** | **Low** | **Optimal (Preserves edges, high throughput)** |

---

## Verified Parameter Counts

The parameter counts across width configurations were empirically verified in `scripts/verify_params.py`:

| Model Configuration | Base Width ($C$) | Stage Configuration | Middle Blocks | Total Parameters | Trainable Parameters | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NAFNet Width 32** | 32 | `[1, 1, 1]` | 1 | 1,129,028 | 1,129,028 | Verified (Baseline) |
| **NAFNet Width 48** | **48** | `[1, 1, 1]` | **1** | **2,521,444** | **2,521,444** | **Verified (Primary Target)** |
| **NAFNet Width 64** | 64 | `[1, 1, 1]` | 1 | 4,465,796 | 4,465,796 | Verified (Capacity Study) |

---

## Computational Complexity (FLOPs)

Theoretical computational complexity was measured using PyTorch `torch.utils.flop_counter.FlopCounterMode` on the primary **NAFNet Width 48** model:

| Input Resolution | Output Resolution | Total FLOPs | GFLOPs | Measurement Method |
| :---: | :---: | :---: | :---: | :--- |
| $128 \times 128$ | $256 \times 256$ | 4,250,760,192 | **4.25 GFLOPs** | PyTorch `FlopCounterMode` |
| $256 \times 256$ | $512 \times 512$ | 17,001,575,424 | **17.00 GFLOPs** | PyTorch `FlopCounterMode` |

> [!NOTE]
> These figures represent mathematical FLOP counts computed by tracking low-level tensor operations. They are not hardware benchmark results (e.g. GPU execution latency).

---

## Code References

- Full Model Implementation: [`src/models/nafnet.py`](file:///d:/Programming/python/semicon/src/models/nafnet.py)
- NAFBlock Primitives: [`src/models/nafblock.py`](file:///d:/Programming/python/semicon/src/models/nafblock.py)
- Foundational Building Blocks (`LayerNorm2d`, `SimpleGate`, `SCA`): [`src/models/blocks.py`](file:///d:/Programming/python/semicon/src/models/blocks.py)
- Model Builder Factory: [`src/models/builder.py`](file:///d:/Programming/python/semicon/src/models/builder.py)
- Parameter Verification Script: [`scripts/verify_params.py`](file:///d:/Programming/python/semicon/scripts/verify_params.py)
