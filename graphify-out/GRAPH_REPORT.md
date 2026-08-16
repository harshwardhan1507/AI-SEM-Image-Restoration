# Graph Report - .  (2026-08-10)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1068 nodes · 2037 edges · 46 communities (42 shown, 4 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 155 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d99ae0b4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Evaluator
- test_losses.py
- NAFBlock
- calculate_psnr
- test_builder.py
- CheckpointManager
- QualitativeEvaluator
- main
- DatasetPair
- Config
- NAFNet
- ConfigDict
- SEMDataset
- build_model
- TestNAFBlock
- test_checkpoint.py
- SEMDatasetAnalyzer
- test_model.py
- DatasetScanner
- _make_trainer
- experiment_tracker.py
- test_inference.py
- tiny_model
- slide_window_inference
- evaluate_qualitative.py
- get_transforms
- .__init__
- evaluate.py
- .load
- setup_logger
- TestSimpleGate
- .__init__
- benchmark_dataset.py
- TestAMPAutocast
- IdentityModel
- .forward
- DummyDataset
- TestGradientClipping
- .__call__
- _calculate_tile_starts
- TestDeterminism
- TestAMPBF16
- create_github_issues.py
- src/__init__.py
- tests/__init__.py
- sem-image-restoration-nafnet

## God Nodes (most connected - your core abstractions)
1. `NAFNet` - 61 edges
2. `CheckpointManager` - 52 edges
3. `Config` - 52 edges
4. `Evaluator` - 43 edges
5. `ExperimentTracker` - 37 edges
6. `NAFBlock` - 36 edges
7. `Trainer` - 32 edges
8. `SEMDataset` - 30 edges
9. `SyntheticSEMTestDataset` - 24 edges
10. `_make_trainer()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `DummyDataset` --uses--> `SEMDataset`  [INFERRED]
  tests/test_builder.py → src/datasets/sem_dataset.py
- `DummyModel` --uses--> `CheckpointManager`  [INFERRED]
  tests/test_inference.py → src/engine/checkpoint.py
- `IdentityModel` --uses--> `CheckpointManager`  [INFERRED]
  tests/test_inference.py → src/engine/checkpoint.py
- `DummyDataset` --uses--> `CheckpointManager`  [INFERRED]
  tests/test_train.py → src/engine/checkpoint.py
- `SyntheticSEMDataset` --uses--> `CheckpointManager`  [INFERRED]
  tests/test_trainer.py → src/engine/checkpoint.py

## Import Cycles
- None detected.

## Communities (46 total, 4 thin omitted)

### Community 0 - "Evaluator"
Cohesion: 0.05
Nodes (53): Figure, Evaluator, Any, DataLoader, device, Module, Path, SummaryWriter (+45 more)

### Community 1 - "test_losses.py"
Cohesion: 0.05
Nodes (53): build_loss(), build_loss_function(), _extract_loss_config(), _get_param(), Any, Module, Loss builder factory module for instantiating loss modules from configuration.…, Construct a loss function module from a configuration object. Args: cfg:… (+45 more)

### Community 2 - "NAFBlock"
Cohesion: 0.06
Nodes (33): LayerNorm2d, Foundational computational primitives for NAFNet architecture. This module…, Custom 2D Layer Normalization operating on BCHW image tensors. Normalizes…, Simplified Channel Attention (SCA) module for NAFNet. SCA simplifies…, Activation-free non-linear interaction via channel splitting and…, SimpleGate, SimplifiedChannelAttention, NAFNet model architecture, building blocks (SimpleGate, SCA, NAFBlock), and… (+25 more)

### Community 3 - "calculate_psnr"
Cohesion: 0.07
Nodes (46): ArrayLike, Evaluation execution engine module for SEM image restoration. This module…, PyTorch Trainer module for SEM image restoration training execution. This…, Execute validation over the validation DataLoader. Computes validation loss,…, Quantitative evaluation metrics calculation modules (PSNR, SSIM, LPIPS)., calculate_lpips(), _get_lpips_model(), device (+38 more)

### Community 4 - "test_builder.py"
Cohesion: 0.07
Nodes (49): build_dataloader(), build_dataloaders(), _detect_splits(), _extract_config_value(), Any, DataLoader, Dataset, Path (+41 more)

### Community 5 - "CheckpointManager"
Cohesion: 0.09
Nodes (33): CheckpointManager, Persistent model checkpoint management module for SEM image restoration. This…, Manager for persistent model checkpoint saving, loading, and best-model…, Training execution engine for SEM image restoration with NAFNet. Orchestrates…, Trainer, ExperimentTracker, Standardized Machine-Readable Experiment Tracker. Tracks metadata,…, Tensor (+25 more)

### Community 6 - "QualitativeEvaluator"
Cohesion: 0.08
Nodes (34): main(), parse_args(), Namespace, Main entry point for qualitative evaluation CLI., Parse command line arguments for qualitative evaluation CLI. Args: args:…, ndarray, Path, Tensor (+26 more)

### Community 7 - "main"
Cohesion: 0.09
Nodes (33): patch, DummyDataset, Dataset, Path, Unit tests for train.py CLI entry point and training orchestration pipeline., Test main() raises FileNotFoundError for non-existent resume path., Test main() orchestration for fresh training execution., Simple mock dataset for training CLI unit tests. (+25 more)

### Community 8 - "DatasetPair"
Cohesion: 0.11
Nodes (28): Exception, DatasetPair, Dataclass holding file path pairings for a dataset sample. Attributes:…, DatasetValidationError, DatasetValidator, InvalidDtypeError, InvalidShapeError, Any (+20 more)

### Community 9 - "Config"
Cohesion: 0.10
Nodes (26): Config, load_config(), Path, YAML Configuration parser and schema validation module for SEM NAFNet…, Recursively update dictionary d with keys from dictionary u. Args: d: Target…, Master Configuration Loader and Schema Manager., Return native dictionary representation of configuration. Returns: Dict[str,…, Save configuration instance to a YAML file. Args: save_path: Destination file… (+18 more)

### Community 10 - "NAFNet"
Cohesion: 0.09
Nodes (19): NAFNet, Tensor, Forward pass through the complete NAFNet architecture. Executes the full…, Pad input tensor to ensure spatial dimensions are divisible by padder_size.…, Complete NAFNet encoder-decoder architecture with 2× PixelShuffle SR tail.…, Tests for direct NAFNet constructor validation., NAFNet constructs successfully with valid parameters., NAFNet raises ValueError for non-positive img_channel. (+11 more)

### Community 11 - "ConfigDict"
Cohesion: 0.07
Nodes (22): dict, ConfigDict, Any, Initialize Config instance with optional base configuration dict. Args:…, Return raw ConfigDict object. Returns: ConfigDict: Dot-accessible configuration…, Forward attribute lookup to underlying ConfigDict. Args: item: Configuration…, Dictionary-style key lookup. Args: item: Key string. Returns: Any: Value…, Dot-accessible recursive dictionary class for configuration management. (+14 more)

### Community 12 - "SEMDataset"
Cohesion: 0.09
Nodes (25): Dataset, Path, Tensor, PyTorch Dataset for paired SEM restoration micrographs., Initialize SEMDataset. Args: root_dir: Path to root dataset directory. split:…, Return total number of dataset samples. Returns: int: Number of items in…, Load, clip, and format a single .npy file into a PyTorch tensor. Args:…, Retrieve a single dataset sample by integer index. Args: index: Dataset sample… (+17 more)

### Community 13 - "build_model"
Cohesion: 0.10
Nodes (21): build_model(), _extract_model_config(), _get_param(), Any, Module, Model builder factory for NAFNet architecture construction. This module…, Extract model sub-configuration from a configuration object. Args: cfg: Full…, Read a parameter from a configuration object with fallback default. Args: cfg:… (+13 more)

### Community 14 - "TestNAFBlock"
Cohesion: 0.07
Nodes (16): Unit tests and verification suite for NAFBlock (nafblock.py)., Verify numerical stability on extreme inputs (small, large, zeros, ones,…, Comprehensive test suite for NAFBlock computational module., Calculate FLOPs for (1, C, H, W) input and log performance benchmark summary., Verify constructor parameter validation., Verify train() and eval() modes yield identical outputs when dropout=0.0., Verify train() mode with dropout > 0 alters output compared to eval() mode., Verify forward pass under PyTorch Automatic Mixed Precision (autocast). (+8 more)

### Community 15 - "test_checkpoint.py"
Cohesion: 0.12
Nodes (25): DummyModel, Path, Tensor, Pytest unit test suite for CheckpointManager (src/engine/checkpoint.py)., Test 4: Advance scheduler, save, load into fresh scheduler (and test None…, Test 5: Save at known epoch, load, verify epoch recovered., Simple model for checkpoint testing., Test 6: Save checkpoint with known best PSNR, load, verify metric recovered. (+17 more)

### Community 16 - "SEMDatasetAnalyzer"
Cohesion: 0.11
Nodes (19): main(), parse_args(), Any, Logger, Namespace, Dataset characterization and analysis tool for SEM image restoration. This…, Validate directory hierarchy and discover GT/NoisyLR folders. Returns:…, Verify paired GT and NoisyLR files and check file integrity. Returns: Dict[str,… (+11 more)

### Community 17 - "test_model.py"
Cohesion: 0.08
Nodes (19): parametrize, fixture, Comprehensive unit tests for NAFNet architecture and model builder. Verifies…, Tests for various batch sizes., Forward pass works with batch sizes 1, 2, and 4., Tests for parameter count validation against analytical formula., Verify NAFBlock parameter count matches formula P(C) = 7C² + 33C., Verify total parameter count for tiny configuration is reasonable. (+11 more)

### Community 18 - "DatasetScanner"
Cohesion: 0.11
Nodes (19): DatasetScanner, Path, Dataset scanner module for SEM image restoration. This module provides…, Directory scanner for indexing paired SEM restoration dataset files., Initialize DatasetScanner with dataset root directory. Args: root_dir: Path to…, Check if file is a valid .npy array file and not hidden metadata. Args: path:…, Collect all valid .npy files within a directory sorted by name. Args:…, Scan a dataset split ('train' or 'test') and build paired file index. Args:… (+11 more)

### Community 19 - "_make_trainer"
Cohesion: 0.15
Nodes (14): _collate_fn(), _make_trainer(), Any, DataLoader, Dataset, skipif, SummaryWriter, BF16 on CUDA: GradScaler must be disabled. (+6 more)

### Community 20 - "experiment_tracker.py"
Cohesion: 0.11
Nodes (20): _detect_platform(), _get_compute_environment(), _get_git_commit(), device, Standardized Reproducible Experiment Tracking Module for SEM Image Restoration.…, Retrieve current Git commit SHA safely. Returns: Optional[str]: 40-character…, Detect current execution platform environment. Returns: str: 'Kaggle', 'Google…, Capture current compute, GPU, PyTorch, and Python runtime information. Args:… (+12 more)

### Community 21 - "test_inference.py"
Cohesion: 0.12
Nodes (21): DummyModel, Unit and integration test suite for sliding-window inference pipeline.…, Verify shape and value contract for same-resolution (upscale=1) inference., Verify shape contract for super-resolution (upscale=2) inference., Verify safe padding and unpadding when input image is smaller than tile size., Verify fallback constant padding for 1x1 image., Verify inference on non-divisible spatial dimensions (e.g. 137x213)., Verify that varying tile_batch_size yields identical outputs. (+13 more)

### Community 22 - "tiny_model"
Cohesion: 0.12
Nodes (13): Tests for tensor shape contracts through the forward pass., Primary shape contract: (B,1,128,128) -> (B,1,256,256)., Shape contract: (B,1,256,256) -> (B,1,512,512)., Shape contract: (B,1,64,64) -> (B,1,128,128)., Shape contract for non-square input: (B,1,128,64) -> (B,1,256,128)., Shape contract for dimensions not divisible by padder_size. Input is auto-…, Forward raises ValueError for incorrect input channel count., Forward raises ValueError for non-4D input tensor. (+5 more)

### Community 23 - "slide_window_inference"
Cohesion: 0.19
Nodes (13): dtype, _generate_gaussian_weights(), device, Module, Tensor, Sliding-window inference engine module for full-resolution SEM image…, Execute sliding-window inference on input image tensor. Args: x: Input image…, Generate a deterministic 2D Gaussian spatial weighting map for tile blending.… (+5 more)

### Community 24 - "evaluate_qualitative.py"
Cohesion: 0.13
Nodes (16): Any, ndarray, Path, CLI script for running qualitative restoration failure analysis on SEM…, Resolve prediction array from verified predictions directory or checkpoint.…, resolve_prediction_or_inference(), infer_nafnet_params_from_state_dict(), load_model_and_weights() (+8 more)

### Community 25 - "get_transforms"
Cohesion: 0.15
Nodes (14): get_transforms(), PairedTransforms, Paired spatial data augmentation module for SEM image restoration. This module…, Construct PairedTransforms instance. Args: is_train: If True, enables spatial…, Synchronized spatial data augmentations for paired SEM micrographs., Initialize PairedTransforms pipeline. Args: is_train: If True, builds random…, Path, Pytest unit test suite for paired data augmentations… (+6 more)

### Community 26 - ".__init__"
Cohesion: 0.15
Nodes (12): _get_config_value(), _normalize_dataset_path(), Any, Module, Optimizer, Path, Extract nested configuration value from Config instance., Update metrics record after a validation epoch and save record incrementally.… (+4 more)

### Community 27 - "evaluate.py"
Cohesion: 0.17
Nodes (14): main(), parse_args(), Namespace, CLI script for running patch-tiling sliding-window inference on SEM…, Resolve device setting to explicit 'cuda' or 'cpu' string., Main execution entry point for evaluate.py., Parse command line arguments for prediction CLI. Args: args: Optional list of…, resolve_device() (+6 more)

### Community 28 - ".load"
Cohesion: 0.17
Nodes (10): Any, device, Module, Optimizer, Path, Load checkpoint state dictionary and restore model/optimizer/scheduler…, Return path to best_model.pth if it exists, else None., Return path to highest-epoch periodic checkpoint file if any exists, else None. (+2 more)

### Community 29 - "setup_logger"
Cohesion: 0.17
Nodes (13): Logger, Path, Centralized logging infrastructure module for SEM NAFNet restoration. This…, Initialize and configure a dual console/file logging instance. Args: name: Name…, setup_logger(), Path, Pytest unit test suite for utility modules (logger.py, seed.py)., Test setup_logger creates log directory and persistent log file. (+5 more)

### Community 30 - "TestSimpleGate"
Cohesion: 0.14
Nodes (8): Test suite for SimpleGate module., Verify SimpleGate has zero learnable parameters., Verify shape contract (B, 2C, H, W) -> (B, C, H, W)., Verify exception handling for non-4D tensors or odd channel counts., Verify element-wise multiplication of split channels., Verify gradient flow through SimpleGate back to input tensor., Verify train() and eval() modes yield identical outputs., TestSimpleGate

### Community 31 - ".__init__"
Cohesion: 0.18
Nodes (8): Any, DataLoader, device, Module, Optimizer, SummaryWriter, Execute a single training epoch. Args: epoch: Current epoch number (1-indexed,…, Execute the full training loop across all epochs. For each epoch: 1. Run…

### Community 32 - "benchmark_dataset.py"
Cohesion: 0.31
Nodes (9): benchmark_dataset(), generate_benchmark_report(), main(), Any, Path, Dataset performance benchmarking and validation script for SEM NAFNet…, Execute dataset benchmarking CLI utility., Run comprehensive performance benchmarks on SEMDataset. Args: dataset_path:… (+1 more)

### Community 33 - "TestAMPAutocast"
Cohesion: 0.22
Nodes (7): skipif, Tests for torch.compile compatibility., Model can be compiled with torch.compile and produces correct shapes. Note:…, Tests for Automatic Mixed Precision compatibility., Forward pass works under CUDA AMP autocast., TestAMPAutocast, TestTorchCompile

### Community 34 - "IdentityModel"
Cohesion: 0.22
Nodes (5): IdentityModel, Tensor, Verify that Gaussian weighted accumulation and normalization accurately…, Identity model for upscale=1 or upscale=2 testing exact intensity preservation., test_seamless_gaussian_blending_normalization()

### Community 35 - ".forward"
Cohesion: 0.29
Nodes (4): Tensor, Forward pass for SimpleGate. Args: x (torch.Tensor): Input tensor of shape (B,…, Forward pass for SimplifiedChannelAttention. Args: x (torch.Tensor): Input…, Forward pass for LayerNorm2d. Args: x (torch.Tensor): Input tensor of shape (B,…

### Community 36 - "DummyDataset"
Cohesion: 0.29
Nodes (4): DummyDataset, Any, Dataset, Simple dummy dataset for testing edge cases.

### Community 37 - "TestGradientClipping"
Cohesion: 0.40
Nodes (3): Test 5: Gradient clipping is applied when grad_clip_norm is set., Verify that after clipping, gradient norms are bounded., TestGradientClipping

### Community 38 - ".__call__"
Cohesion: 0.50
Nodes (3): ndarray, Tensor, Apply synchronized spatial transformations to input and target images. Args:…

### Community 39 - "_calculate_tile_starts"
Cohesion: 0.50
Nodes (4): _calculate_tile_starts(), Calculate 1D tile start coordinates guaranteeing 100% boundary coverage. Args:…, Verify tile start coordinate calculation for various dimensions., test_tile_starts_calculation()

### Community 40 - "TestDeterminism"
Cohesion: 0.50
Nodes (3): Tests for deterministic output given identical inputs., Identical inputs produce identical outputs in eval mode., TestDeterminism

### Community 41 - "TestAMPBF16"
Cohesion: 0.50
Nodes (3): Test 8: BF16 autocast without GradScaler., BF16 on CPU: autocast may be supported, GradScaler must be disabled., TestAMPBF16

## Knowledge Gaps
- **1 isolated node(s):** `sem-image-restoration-nafnet`
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NAFNet` connect `NAFNet` to `TestAMPAutocast`, `NAFBlock`, `IdentityModel`, `TestDeterminism`, `build_model`, `test_model.py`, `test_inference.py`, `tiny_model`, `evaluate_qualitative.py`, `evaluate.py`?**
  _High betweenness centrality (0.230) - this node is a cross-community bridge._
- **Why does `CheckpointManager` connect `CheckpointManager` to `IdentityModel`, `calculate_psnr`, `TestGradientClipping`, `main`, `TestAMPBF16`, `Config`, `test_checkpoint.py`, `_make_trainer`, `test_inference.py`, `slide_window_inference`, `evaluate.py`, `.load`, `.__init__`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `Evaluator` connect `Evaluator` to `calculate_psnr`, `slide_window_inference`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `NAFNet` (e.g. with `NAFBlock` and `DummyModel`) actually correct?**
  _`NAFNet` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `CheckpointManager` (e.g. with `Trainer` and `DummyModel`) actually correct?**
  _`CheckpointManager` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `Config` (e.g. with `ExperimentTracker` and `DummyDataset`) actually correct?**
  _`Config` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `Evaluator` (e.g. with `SyntheticSEMTestDataset` and `TestComparisonImageGeneration`) actually correct?**
  _`Evaluator` has 17 INFERRED edges - model-reasoned connections that need verification._