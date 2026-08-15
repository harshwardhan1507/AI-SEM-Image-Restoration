---
name: auto-documenter
description: Automatically capture test execution logs, benchmark metrics, and experiment results, then generate or update project documentation reports and README.md.
---

# Automated Experiment & Test Documentation Skill

This skill automates the workflow of extracting test, benchmark, and experiment outputs and keeping repository documentation continuously up to date.

## When to Trigger
Use this skill whenever:
- Pytest test runs (`pytest tests/`) complete and emit results.
- Dataset or model benchmarks (`scripts/benchmark_dataset.py`, `scripts/verify_params.py`) are executed.
- Training or evaluation runs (`train.py`, `scripts/evaluate_qualitative.py`) produce new metrics or visual artifacts.
- The user requests to "document recent test results", "update benchmark docs", or "log experiment findings".

---

## Workflow Steps

### 1. Locate and Extract Execution Logs & Artifacts
Inspect the following locations for runtime outputs:
- **Pytest Results**: Execution summaries from recent `pytest` runs.
- **Benchmark Reports**: Log files and tables in `results/benchmarks/` and `results/tables/`.
- **Qualitative Images**: Visual grid artifacts in `results/images/qualitative_analysis/`.
- **Experiment Configurations**: YAML configs in `configs/experiments/`.
- **TensorBoard / Checkpoints**: Model weights and event logs in `experiments/checkpoints/` or `outputs/`.

### 2. Extract Key Performance Metrics
Identify and summarize quantitative metrics:
- **Image Quality**: PSNR (dB), SSIM, Charbonnier loss values.
- **Compute Efficiency**: Model parameter count, FLOPs/MACs, inference latency (ms/MPixel), RAM allocation (MB), loader throughput (samples/sec).
- **Test Integrity**: Number of passed/failed unit tests, code coverage metrics.

### 3. Generate or Update Verification Reports
Depending on the type of run, update or create markdown files in `docs/`:

- **For Unit/Integration Tests**:
  Create or update `docs/verification/<test_name>_report.md` detailing:
  - Date & environment setup.
  - Test suite status (pass/fail summary).
  - Component integrity assertions verified.

- **For Capacity & Model Experiments**:
  Create or update `docs/experiments/<exp_id>_report.md` detailing:
  - Experiment ID, hyperparameters, and width/depth settings.
  - Quantitative comparison against baseline (PSNR/SSIM delta).
  - Quality-vs-compute tradeoff conclusions.

- **For Dataset Benchmarks**:
  Update `results/benchmarks/dataset_benchmark_report.md` with:
  - Measured latency, throughput, and memory consumption.

### 4. Sync Benchmarks with Master README
If new benchmark records or SOTA metrics are achieved:
- Update the **Empirical Capacity Scaling Benchmarks** table in [README.md](file:///c:/AI-SEM-Image-Restoration-main/README.md).
- Update qualitative evaluation image references if new high-quality grids were generated in `results/images/qualitative_analysis/`.

---

## Output Quality Standards
- **No Emojis**: Maintain a clean, academic, research-grade Markdown formatting standard.
- **Exact Empirical Data**: Never estimate or hallucinate numbers; rely strictly on log data.
- **File Links**: Use relative Markdown links for repository files (e.g. `[exp002.yaml](configs/experiments/exp002_nafnet_width48.yaml)`).
