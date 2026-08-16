# KLA Webinar — Key Findings & Optimal Solution Strategy

> **Source:** KLA / Semicon India Hackathon technical webinar  
> **Date:** 2026-08-08  
> **Purpose:** Capture KLA's confirmed requirements, recommendations, and strategic implications for the SEM image restoration project.

---

## 1. Executive Summary

KLA clarified that the task is **AI-based restoration of degraded SEM inspection images**.

The benchmark degradation pipeline consists of three known mechanisms:

1. **Gaussian noise** — additive
2. **Speckle noise** — multiplicative
3. **Downsampling** — resolution degradation

The final solution is evaluated on three major axes:

- **Restoration quality**
- **End-to-end inference efficiency**
- **Training / compute hygiene**

The key strategic lesson is:

> **Do not optimize for maximum model size alone. Optimize for the best restoration-quality / inference-efficiency trade-off while maintaining reproducible, well-documented training.**

Our current NAFNet baseline achieved **29.4118 dB PSNR** and **0.7891 SSIM**, with a **+6.5049 dB improvement** over the measured raw noisy baseline.

---

## 2. Confirmed Problem Characteristics

### Degradation Types

KLA explicitly identified:

- Gaussian noise
- Speckle noise
- Downsampling

Gaussian noise is additive; speckle noise is multiplicative.

The degradation mechanisms may occur in different orders. Explicitly detecting the order is **not required**.

A single-shot model is therefore valid:

```text
Noisy + Low Resolution
          ↓
       NAFNet
          ↓
Clean + High Resolution
```

---

## 3. Hidden Test Set & Generalization

The hidden evaluation set:

- Uses the same general degradation mechanisms.
- May have somewhat different noise severity levels.
- Can contain substantially different image content and structures.
- Includes both in-domain and out-of-domain image content.

The major distribution shift is expected to come primarily from **image content / structures**, rather than completely unseen degradation mechanisms.

### Implication

Prioritize:

- appropriate augmentation
- robust restoration
- sufficient model capacity
- held-out validation
- visual inspection

---

## 4. Evaluation Metrics

KLA confirmed that final restoration quality uses:

- **PSNR**
- **SSIM**
- **LPIPS**

The final ranking uses a **fixed weighted combination**, with exact weights undisclosed.

Therefore, PSNR alone is not enough.

Our experiment tracking should eventually contain:

```text
PSNR
SSIM
LPIPS
Inference latency
Visual results
```

---

## 5. Current Baseline

### Experiment

`exp001_nafnet_baseline`

### Configuration

```text
Model:
NAFNet
width = 32
enc_blocks = [1,1,1]
mid_blocks = 1
dec_blocks = [1,1,1]
upscale = 2

Loss:
CharbonnierLoss
eps = 1e-3

Optimizer:
AdamW
lr = 1e-3
weight_decay = 1e-4

Scheduler:
CosineAnnealingLR
T_max = 50
eta_min = 1e-6

Batch size = 4
Epochs = 50
AMP = False
```

### Results

```text
Raw noisy baseline PSNR:       22.9069 dB
Best NAFNet validation PSNR:   29.4118 dB
Best NAFNet validation SSIM:    0.7891
Best PSNR epoch:                50
Best SSIM epoch:                46

PSNR improvement:              +6.5049 dB
Required improvement:          +3.0000 dB

Result: PASS
```

The baseline proves that the complete training/evaluation pipeline works and provides substantial restoration improvement.

---

## 6. Model Scaling Strategy

KLA recommended experimenting with model capacity and stopping when additional capacity stops producing meaningful gains.

### Recommended progression

#### Stage 1 — Baseline

```text
width = 32
enc = [1,1,1]
mid = 1
dec = [1,1,1]
```

Reference:

```text
PSNR = 29.4118 dB
SSIM = 0.7891
```

#### Stage 2 — Moderate scaling

A reasonable first experiment:

```text
width = 48
enc = [2,2,2]
mid = 2
dec = [2,2,2]
```

Treat this as an experiment, not a predetermined final architecture.

#### Stage 3 — Larger scaling

Only proceed if Stage 2 produces a meaningful improvement.

### Decision rule

```text
Larger model → meaningful improvement
        ↓
Continue exploring capacity

Larger model → negligible improvement
        ↓
Stop scaling
        ↓
Investigate loss / augmentation / preprocessing
```

### Important

Do **not** blindly jump to a ~68M parameter model.

Larger models increase:

- training time
- GPU memory
- inference latency
- deployment complexity

KLA's preferred direction is a strong quality/efficiency trade-off.

---

## 7. Data Augmentation & Synthetic Degradation

KLA explicitly allows generating additional synthetic training pairs from ground-truth images.

Potential pipeline:

```text
Ground Truth
     │
     ├── Gaussian noise
     ├── Speckle noise
     ├── Downsampling
     └── Different combinations / severity
              ↓
       Synthetic degraded pairs
```

However, KLA also said synthetic degradation can help or hurt.

Therefore:

> Treat synthetic degradation as an experiment and validate it on held-out data before adopting it.

---

## 8. OOD Generalization

KLA recommended augmentation as one route to better OOD performance.

The goal is to avoid memorizing exact training structures.

Useful areas to investigate:

- image augmentation
- degradation-aware augmentation
- different degradation severity
- synthetic degraded pairs

---

## 9. Loss Function Strategy

KLA explicitly stated that loss selection is part of the challenge.

They mentioned common starting points such as:

- MSE
- L1
- SSIM
- combinations / task-specific losses

Our current Charbonnier loss is a valid baseline.

Possible experiment sequence:

```text
Exp001: Charbonnier
   ↓
Exp002: L1 / MSE comparison
   ↓
Exp003: Charbonnier + structural loss
   ↓
Exp004: Pixel + frequency/structural objective
```

Only retain changes that improve held-out performance.

---

## 10. Frequency-Domain Training

KLA confirmed that frequency-domain training is a valid direction.

Potential experiment:

```text
Spatial-domain restoration
          +
Frequency-domain supervision
```

KLA did not claim it will definitely improve performance, so it should remain an experiment rather than an assumption.

---

## 11. External Data & Pretrained Models

KLA explicitly permits:

- publicly available datasets
- publicly available pretrained model weights
- domain adaptation
- fine-tuning pretrained restoration models

provided licensing permits use in the hackathon.

If external resources are used, document:

### Dataset

- Name
- Link
- License
- Research paper

### Model / weights

- Model name
- Source / model card
- Link
- License where applicable

---

## 12. Visual Evaluation

KLA strongly recommended inspecting actual restored images rather than relying only on metrics.

For important experiments compare:

```text
NoisyLR
   ↓
Prediction
   ↓
Ground Truth
```

Look for:

- remaining Gaussian noise
- remaining speckle noise
- blurred structures
- lost high-frequency detail
- oversmoothing
- hallucinated structures
- ringing
- reconstruction artifacts
- tiling seams

A metric improvement does not automatically mean the visual restoration is better.

---

## 13. Inference Requirements

KLA expects the final submission to provide an inference script that:

1. Reads images from an input directory.
2. Runs restoration.
3. Writes restored images to an output directory.

Recommended interface:

```bash
python scripts/evaluate.py     --input_dir <input_directory>     --output_dir <output_directory>
```

Pipeline:

```text
Input directory
      ↓
Image loading
      ↓
Pre-processing
      ↓
Batching / tiling
      ↓
GPU inference
      ↓
Post-processing
      ↓
Output saving
```

---

## 14. End-to-End Inference Speed

KLA clarified that inference timing includes:

```text
Disk read
   ↓
Pre-processing
   ↓
GPU transfer
   ↓
Model forward
   ↓
Post-processing
   ↓
Disk write
```

It is not just neural-network forward-pass time.

Therefore, inference engineering is part of the competition.

---

## 15. Batch Processing

KLA prefers batch processing where GPU memory allows it.

Prefer:

```text
Images
  ↓
Batch
  ↓
GPU
  ↓
Restoration
  ↓
Outputs
```

rather than processing every image independently.

Batch size should be chosen according to available GPU memory and measured throughput.

---

## 16. Sliding-Window / Tiled Inference

Large images can be processed using overlapping tiles:

```text
Large image
     ↓
Overlapping tiles
     ↓
NAFNet
     ↓
Weighted blending
     ↓
Full-resolution output
```

KLA confirmed that the time required for all tiles counts toward inference latency.

Therefore:

```text
More overlap
    ↓
Better seam reduction
    ↓
More computation
    ↓
Higher latency
```

Tiling should be benchmarked rather than assumed to be free.

KLA indicated hidden images are expected to be approximately:

```text
256 × 256
512 × 512
```

---

## 17. Output Range / Post-processing

KLA stated that they will **not perform clipping or normalization before scoring**.

Our inference pipeline must therefore produce valid final outputs.

```text
Model output
     ↓
Correct range / post-processing
     ↓
Image serialization
     ↓
Saved restoration
```

Do not rely on KLA's evaluator to fix invalid output ranges.

---

## 18. Reproducibility & Training Hygiene

KLA considers training hygiene as part of evaluation.

The final project should provide:

```text
Training code
Configuration files
Experiment definitions
Checkpoint
Dependencies / environment specification
Reproducible configuration
Inference script
Documentation
```

Important practices:

- reproducible seeds
- clear configuration
- checkpointing
- experiment tracking
- documented dependencies
- clear training commands
- documented external resources

---

## 19. Recommended Experiment Tracking

Every meaningful experiment should record:

```text
Experiment ID
Model architecture
Parameter count
Dataset
Augmentations
Loss
Optimizer
Learning rate
Scheduler
Epochs
Batch size

Best PSNR
Best SSIM
Best LPIPS
Best epoch

Inference latency
Throughput

Visual observations
Failure cases
Decision
```

Example:

```text
exp002_nafnet_scaled

PSNR:      XX.XXXX dB
SSIM:      X.XXXX
LPIPS:     X.XXXX
Params:    XX M
Latency:   XX ms/image

Decision:
Accepted / Rejected

Reason:
...
```

---

## 20. Recommended Development Order

Based on KLA's guidance and our current project state:

```text
Baseline
   ↓
Complete inference pipeline
   ↓
Add LPIPS
   ↓
Moderate model scaling
   ↓
Evaluate quality + latency
   ↓
Loss / augmentation experiments
   ↓
Visual failure analysis
   ↓
Select best quality/speed trade-off
   ↓
Final training
   ↓
Final inference benchmark
   ↓
Reproducible packaging
```

---

## 21. Priority Matrix

### 🔴 Highest Priority

1. Finish production inference pipeline.
2. Add LPIPS evaluation.
3. Run a moderate NAFNet scaling experiment.
4. Inspect restored images visually.
5. Track PSNR + SSIM + LPIPS + inference speed.
6. Preserve reproducibility and experiment configuration.

### 🟡 Medium Priority

7. Loss-function experiments.
8. Synthetic degradation / augmentation.
9. Frequency-domain supervision.
10. Further model scaling.
11. Inference optimization.

### 🟢 Optional / Later

12. External datasets.
13. Pretrained restoration weights.
14. Degradation-specific architectural components.
15. CUDA-specific optimizations.

### ❌ Not Required

- Explicit degradation-order detection.
- Defect-aware loss for the provided dataset.
- Supporting arbitrary degradation mechanisms.
- Blindly maximizing parameter count.

---

## 22. Recommended Final Strategy

```text
1. Establish reliable baseline
          ↓
2. Build complete inference pipeline
          ↓
3. Add LPIPS
          ↓
4. Moderate model scaling
          ↓
5. Evaluate quality + latency
          ↓
6. Improve loss / augmentation
          ↓
7. Visual failure analysis
          ↓
8. Select best quality/speed trade-off
          ↓
9. Final training
          ↓
10. Reproducible packaging
```

The objective is **not**:

```text
Largest NAFNet possible
```

The objective is:

```text
Highest practical restoration quality
+
Strong OOD generalization
+
Low end-to-end inference latency
+
Reproducible training
+
Clean final submission
```

---

## 23. Key Takeaways

KLA explicitly confirmed:

- The benchmark focuses on **Gaussian noise, speckle noise, and downsampling**.
- Degradation order does not need to be explicitly detected.
- Public datasets and pretrained models are allowed if properly licensed.
- Synthetic degraded training data is allowed.
- Frequency-domain approaches are allowed.
- Loss selection is part of the challenge.
- Augmentation can help OOD generalization.
- Final evaluation uses **PSNR + SSIM + LPIPS** with undisclosed fixed weights.
- End-to-end inference speed matters.
- Batch inference is preferred where memory allows.
- Tiling time counts toward inference latency.
- Output clipping/normalization is not performed by KLA.
- Reproducible training and inference are required.
- Visual inspection of outputs is strongly recommended.
- A smaller, efficient model with strong quality is preferable to an unnecessarily large model.

### Strategic conclusion

Our baseline is **good enough to establish the pipeline but not necessarily good enough to be the final model**.

The next optimization should be empirical and incremental:

> **Measure → change one major factor → train → evaluate → inspect → compare → keep/reject.**

This provides an evidence-based path toward the final submission instead of blindly increasing model size or adding techniques without measurement.
