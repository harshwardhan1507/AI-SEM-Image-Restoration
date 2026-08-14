# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Machine-readable quantitative results CSV table (`results/tables/quantitative_results.csv`) summarizing baseline, bicubic interpolation, and NAFNet capacity scaling experiments (Width 32 vs 48 vs 64).
- Comprehensive comparative results summary report (`results/tables/results_summary.md`) detailing quality gains (+7.08 dB PSNR over raw input), capacity diminishing returns analysis (confirming `width=48` as optimal deployment choice), dataset throughput metrics (3.546 ms P95 latency), and KLA compliance.
- Master research literature analysis report (`docs/research/research_literature_analysis.md`) for Issue #45.
- Research analysis documents for reproducibility (#54) and external datasets/pretrained models (#55).
- Workspace Skill `.agents/skills/auto-documenter/SKILL.md` for automated experiment, benchmark, and test logging.

### Changed
- Comprehensive update to `README.md` incorporating KLA Hackathon problem physics, embedded dataset visuals, NAFNet architecture diagrams, and empirical capacity scaling benchmarks.

## [0.1.0] - 2026-08-06

### Added
- Complete research-grade repository foundation and directory structure (`src/` package layout).
- Base configurations (`configs/default.yaml`, `train.yaml`, `model.yaml`, `inference.yaml`, `configs/experiments/`).
- Placeholder test suite structure in `tests/`.
- Pre-commit configuration (`.pre-commit-config.yaml`) and pyproject tool settings.
- Environment template (`.env.example`) and unpinned dependency definitions.
- Asset, figure, weight, experiment, log, and output directory structure.

