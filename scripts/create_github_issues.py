"""Helper script to create all GitHub issues defined in project_roadmap.md via gh CLI."""

import json
import subprocess
from typing import Any, Dict, List

REPO = "harshwardhan1507/AI-SEM-Image-Restoration"

ISSUES: List[Dict[str, Any]] = [
    {
        "title": "feat(utils): Implement dual console/file logger and deterministic seed manager",
        "body": """### Objective
Create `src/utils/logger.py` and `src/utils/seed.py` for structured logging and random seed control.

### Scope
Support dual console (INFO) and file (DEBUG) logging. Implement `set_seed(42)` enforcing determinism across Python, NumPy, and PyTorch (CPU/CUDA/CUDNN).

### Deliverables
- `src/utils/logger.py`
- `src/utils/seed.py`

### Acceptance Criteria
- `set_seed(42)` enforces deterministic operations.
- Structured logging format with timestamps.

Dependencies: #1""",
        "milestone": "Milestone 1: Core Infrastructure",
        "labels": ["architecture", "testing", "high priority"],
    },
    {
        "title": "feat(dataset): Implement memory-mapped SEM dataset class",
        "body": """### Objective
Create `src/datasets/sem_dataset.py` implementing `torch.utils.data.Dataset` for `.npy` image pairs.

### Scope
Load `.npy` arrays using `mmap_mode='r'`, filter out macOS `__MACOSX` / `._*` files, validate paired filenames, clip intensity range `[-0.27, 1.93]` to `[0.0, 1.0]`, and return formatted dictionary samples.

### Deliverables
- `src/datasets/sem_dataset.py`
- `tests/test_dataset.py`

### Acceptance Criteria
- Pairs 3200 training items (128x128 NoisyLR vs 256x256 GT).
- Returns dict with `input` (1,128,128) and `target` (1,256,256) tensors.

Dependencies: #1, #2""",
        "milestone": "Milestone 2: Dataset Layer",
        "labels": ["dataset", "performance", "high priority"],
    },
    {
        "title": "feat(dataset): Implement paired spatial data augmentation pipeline",
        "body": """### Objective
Create `src/datasets/transforms.py` applying synchronized spatial transformations.

### Scope
Implement horizontal flips, vertical flips, and 90deg orthogonal rotations using `albumentations`.

### Deliverables
- `src/datasets/transforms.py`

### Acceptance Criteria
- Transformations applied identically to input and target tensors without sub-pixel spatial interpolation blur.

Dependencies: #3""",
        "milestone": "Milestone 2: Dataset Layer",
        "labels": ["dataset", "research"],
    },
    {
        "title": "feat(dataset): Implement DataLoader builder with host memory pinning",
        "body": """### Objective
Create `src/datasets/builder.py` constructing optimized PyTorch DataLoaders.

### Scope
Support `pin_memory=True`, `persistent_workers=True`, configurable worker counts, and deterministic mini-batch collation.

### Deliverables
- `src/datasets/builder.py`

### Acceptance Criteria
- Instantiates train, validation, and test DataLoader objects driven by YAML config.

Dependencies: #3, #4""",
        "milestone": "Milestone 2: Dataset Layer",
        "labels": ["dataset", "performance"],
    },
    {
        "title": "feat(metrics): Implement PSNR and SSIM evaluation modules",
        "body": """### Objective
Create `src/metrics/psnr_ssim.py` for full-reference evaluation of restored micrographs.

### Scope
Implement PSNR and SSIM functions supporting batch tensor inputs and single 2D arrays.

### Deliverables
- `src/metrics/psnr_ssim.py`
- `tests/test_metrics.py`

### Acceptance Criteria
- Metric outputs match SciPy / scikit-image references within 1e-4 tolerance.

Dependencies: #1""",
        "milestone": "Milestone 3: Metrics & Loss Infrastructure",
        "labels": ["metrics", "research", "high priority"],
    },
    {
        "title": "feat(losses): Implement Charbonnier and PSNR loss functions",
        "body": """### Objective
Create `src/losses/charbonnier.py` and `src/losses/psnr_loss.py` for network optimization.

### Scope
Implement Charbonnier Loss (`sqrt(||x-y||^2 + eps^2)`) and differentiable PSNR loss.

### Deliverables
- `src/losses/charbonnier.py`
- `src/losses/psnr_loss.py`
- `src/losses/builder.py`

### Acceptance Criteria
- Differentiable with clean backpropagation autograd gradients.

Dependencies: #1""",
        "milestone": "Milestone 3: Metrics & Loss Infrastructure",
        "labels": ["model", "training"],
    },
    {
        "title": "feat(models): Implement SimpleGate activation and SCA attention modules",
        "body": """### Objective
Create elementary block primitives in `src/models/blocks.py`.

### Scope
Implement SimpleGate (`X1 * X2`) and SimplifiedChannelAttention (Global Avg Pool + Channel Scale).

### Deliverables
- `src/models/blocks.py`

### Acceptance Criteria
- SimpleGate halves channel dimensions (`2C -> C`).
- Zero nonlinear activation functions used.

Dependencies: #1""",
        "milestone": "Milestone 4: NAFNet Model Architecture",
        "labels": ["model", "architecture", "high priority"],
    },
    {
        "title": "feat(models): Implement foundational NAFBlock residual layer",
        "body": """### Objective
Combine Depthwise Conv, LayerNorm, SimpleGate, and SCA into NAFBlock in `src/models/nafblock.py`.

### Scope
Implement residual block with LayerNorm, 3x3 Depthwise Conv, SimpleGate, SCA, and Dropout/DropPath.

### Deliverables
- `src/models/nafblock.py`

### Acceptance Criteria
- Input and output tensor shapes match exactly `(B,C,H,W) -> (B,C,H,W)`.

Dependencies: #8""",
        "milestone": "Milestone 4: NAFNet Model Architecture",
        "labels": ["model", "architecture"],
    },
    {
        "title": "feat(models): Implement complete NAFNet architecture with 2x PixelShuffle upsampling",
        "body": """### Objective
Build full NAFNet model in `src/models/nafnet.py` supporting 2x super-resolution.

### Scope
Construct Head Conv, Encoder, Downsampling, Bottleneck, Decoder with skip connections, 2x PixelShuffle upsampling tail, and Tail Conv.

### Deliverables
- `src/models/nafnet.py`
- `src/models/builder.py`
- `tests/test_model.py`

### Acceptance Criteria
- Given input `(B,1,128,128)`, produces output `(B,1,256,256)`.

Dependencies: #9""",
        "milestone": "Milestone 4: NAFNet Model Architecture",
        "labels": ["model", "architecture", "high priority"],
    },
    {
        "title": "feat(engine): Implement CheckpointManager for model state save/resume",
        "body": """### Objective
Create `src/engine/checkpoint.py` for persistent model checkpointing.

### Scope
Save best model (`best_model.pth`) based on peak validation PSNR, save periodic state, load optimizer/scheduler/epoch states.

### Deliverables
- `src/engine/checkpoint.py`

### Acceptance Criteria
- Saves full state dict (epoch, model, optimizer, scheduler, best metric).

Dependencies: #1, #2""",
        "milestone": "Milestone 5: Training Infrastructure",
        "labels": ["engine", "training"],
    },
    {
        "title": "feat(engine): Implement PyTorch Trainer class with AMP and Cosine LR Scheduler",
        "body": """### Objective
Create `src/engine/trainer.py` to orchestrate training execution loops.

### Scope
Support AdamW optimizer, Cosine Annealing scheduler, PyTorch AMP (`torch.cuda.amp.autocast`), gradient clipping, TensorBoard logging.

### Deliverables
- `src/engine/trainer.py`

### Acceptance Criteria
- Executes training epoch loops cleanly without CUDA memory leaks.
- AMP FP16/BF16 reduces VRAM memory consumption by ~40%.

Dependencies: #5, #7, #10, #11""",
        "milestone": "Milestone 5: Training Infrastructure",
        "labels": ["engine", "training", "performance", "high priority"],
    },
    {
        "title": "feat(engine): Implement Evaluator class and visual grid generator",
        "body": """### Objective
Create `src/engine/evaluator.py` for validation evaluation and visual grid logging.

### Scope
Compute mean dataset PSNR and SSIM scores, generate visual triple grids (Input vs Prediction vs Target), and log error difference maps.

### Deliverables
- `src/engine/evaluator.py`

### Acceptance Criteria
- Logs visual comparison figures to TensorBoard and `outputs/predictions/`.

Dependencies: #6, #12""",
        "milestone": "Milestone 5: Training Infrastructure",
        "labels": ["engine", "metrics", "documentation"],
    },
    {
        "title": "feat(cli): Integrate train.py entry point script with argparse and config parser",
        "body": """### Objective
Finalize `train.py` root script to parse CLI flags (`--config`, `--experiment`, `--resume`).

### Scope
Parse flags, initialize logger, set random seeds, instantiate DataLoader, build NAFNet model, construct Trainer, initiate training.

### Deliverables
- `train.py`

### Acceptance Criteria
- Command `python train.py --config configs/train.yaml` runs training loop end-to-end.

Dependencies: #12, #13""",
        "milestone": "Milestone 6: Execution & Model Training",
        "labels": ["architecture", "training", "high priority"],
    },
    {
        "title": "research(experiment): Execute baseline experiment exp001 and verify convergence",
        "body": """### Objective
Execute initial training run (`exp001_nafnet_baseline`) on dataset.

### Scope
Train NAFNet model for 50 epochs, monitor loss decay, track PSNR/SSIM improvements, log results.

### Deliverables
- `configs/experiments/exp001.yaml`
- `experiments/exp001_baseline_report.md`

### Acceptance Criteria
- Validation PSNR exceeds raw noisy image PSNR baseline by >= 3.0 dB.

Dependencies: #14""",
        "milestone": "Milestone 6: Execution & Model Training",
        "labels": ["research", "training", "performance"],
    },
    {
        "title": "feat(inference): Implement patch-tiling inference pipeline for large micrographs",
        "body": """### Objective
Create `src/engine/inference.py` supporting sliding-window inference with overlapping tile blending.

### Scope
Divide large input images into overlapping patches, run NAFNet forward inference, apply Gaussian spatial weighting to eliminate seam artifacts.

### Deliverables
- `src/engine/inference.py`
- `scripts/predict.py`

### Acceptance Criteria
- Restores full-resolution micrographs without spatial boundary seam artifacts.

Dependencies: #10""",
        "milestone": "Milestone 7: Inference Pipeline",
        "labels": ["inference", "performance", "high priority"],
    },
    {
        "title": "perf(profiling): Profile CUDA kernel performance and DataLoader throughput",
        "body": """### Objective
Use `torch.profiler` to identify execution bottlenecks.

### Scope
Profile CPU data loading time vs GPU CUDA kernel execution time; optimize worker counts and prefetch factors.

### Deliverables
- `reports/performance_profiling_report.md`

### Acceptance Criteria
- GPU compute utilization exceeds >= 85% during training loops.

Dependencies: #15""",
        "milestone": "Milestone 8: Optimization & Profiling",
        "labels": ["performance", "testing"],
    },
    {
        "title": "test(qa): Implement end-to-end integration and shape assertion test suite",
        "body": """### Objective
Complete `tests/` directory with comprehensive unit tests.

### Scope
Test dataset loading, spatial transformations, metric accuracy, and model forward shape assertions.

### Deliverables
- `tests/test_dataset.py`
- `tests/test_metrics.py`
- `tests/test_model.py`
- `tests/test_pipeline.py`

### Acceptance Criteria
- `pytest` suite passes with 100% success rate.

Dependencies: #3, #6, #10""",
        "milestone": "Milestone 9: Quality Assurance & Testing Suite",
        "labels": ["testing", "high priority"],
    },
    {
        "title": "docs(results): Generate comparative metric tables and final release report",
        "body": """### Objective
Summarize experimental findings, populate result tables, draft release notes.

### Scope
Populate `results/tables/quantitative_results.csv` and `results/tables/results_summary.md` with PSNR, SSIM, and latency benchmarks.

### Deliverables
- `results/tables/results_summary.md`
- `CHANGELOG.md`

### Acceptance Criteria
- Includes final PSNR/SSIM score matrices comparing baseline vs NAFNet.

Dependencies: #15, #16""",
        "milestone": "Milestone 10: Documentation & Final Release",
        "labels": ["documentation", "metrics", "research"],
    },
]


def main() -> None:
    res = subprocess.run(
        ["gh", "issue", "list", "--state", "all", "--json", "title"],
        capture_output=True,
        text=True,
    )
    existing_titles = set()
    if res.returncode == 0 and res.stdout:
        existing_titles = {i["title"] for i in json.loads(res.stdout)}

    for item in ISSUES:
        if item["title"] not in existing_titles:
            cmd = [
                "gh",
                "issue",
                "create",
                "--title",
                item["title"],
                "--body",
                item["body"],
                "--milestone",
                item["milestone"],
            ]
            for lbl in item["labels"]:
                cmd.extend(["--label", lbl])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"Created: {item['title']}")
            else:
                print(f"Error creating {item['title']}: {r.stderr}")
        else:
            print(f"Exists: {item['title']}")


if __name__ == "__main__":
    main()
