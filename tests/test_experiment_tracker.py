"""Unit tests for standardized experiment tracking module (src.utils.experiment_tracker)."""

import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import Config
from src.utils.experiment_tracker import (
    ExperimentTracker,
    _detect_platform,
    _get_compute_environment,
    _get_git_commit,
)


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc(self.conv(x))


def test_git_commit_and_platform_detection():
    """Verify git commit retrieval safely handles execution and platform detection."""
    commit = _get_git_commit()
    # In git repository, commit is either a 40-char string or None
    if commit is not None:
        assert isinstance(commit, str)
        assert len(commit) == 40

    platform_str = _detect_platform()
    assert isinstance(platform_str, str)
    assert len(platform_str) > 0


def test_compute_environment_metadata():
    """Verify compute environment data extraction."""
    compute_info = _get_compute_environment(device_str="cpu")
    assert "platform" in compute_info
    assert compute_info["device"] == "cpu"
    assert "gpu" in compute_info
    assert "pytorch_version" in compute_info
    assert "python_version" in compute_info


def test_experiment_tracker_initialization(tmp_path):
    """Verify ExperimentTracker captures model parameters, config, and creates record file."""
    config_data = {
        "experiment_id": "test_exp_001",
        "system": {"seed": 1234, "device": "cpu"},
        "data": {"dataset_dir": "./test_data", "splits": ["train", "val"]},
        "model": {"name": "SimpleTestModel", "channels": 16},
        "train": {"epochs": 10, "batch_size": 2, "learning_rate": 1e-3},
        "loss": {"name": "L1Loss"},
    }
    config = Config(config_data)
    model = SimpleTestModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.L1Loss()

    tracker = ExperimentTracker(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        record_dir=tmp_path,
    )

    rec = tracker.to_dict()
    assert rec["experiment"]["id"] == "test_exp_001"
    assert rec["model"]["architecture"] == "SimpleTestModel"

    # Verify parameter count calculation
    total_expected = sum(p.numel() for p in model.parameters())
    assert rec["model"]["parameters"]["total"] == total_expected
    assert rec["model"]["parameters"]["trainable"] == total_expected

    # Verify hyperparameter extraction
    assert rec["training"]["optimizer"] == "AdamW"
    assert rec["training"]["learning_rate"] == 1e-3
    assert rec["training"]["scheduler"] == "CosineAnnealingLR"
    assert rec["training"]["loss"] == "L1Loss"
    assert rec["training"]["seed"] == 1234

    # Verify YAML file creation
    record_file = tmp_path / "test_exp_001_record.yaml"
    assert record_file.exists()

    with open(record_file, "r", encoding="utf-8") as f:
        loaded_yaml = yaml.safe_load(f)
    assert loaded_yaml["experiment"]["id"] == "test_exp_001"


def test_experiment_tracker_validation_updates_and_best_epochs(tmp_path):
    """Verify metrics updates, best epoch tracking (PSNR max, SSIM max, LPIPS min), and incremental saving."""
    config = Config({"experiment_id": "metrics_exp"})
    tracker = ExperimentTracker(config=config, record_dir=tmp_path)

    # Initial state should be null / None for all metrics
    initial_rec = tracker.to_dict()
    assert initial_rec["metrics"]["psnr"]["best"] is None
    assert initial_rec["metrics"]["psnr"]["epoch"] is None
    assert initial_rec["metrics"]["ssim"]["best"] is None
    assert initial_rec["metrics"]["lpips"]["best"] is None

    # Epoch 1 validation: PSNR=25.0, SSIM=0.70, LPIPS=0.15
    tracker.update_validation(
        epoch=1,
        val_metrics={"val_psnr": 25.0, "val_ssim": 0.70, "val_lpips": 0.15},
    )
    rec1 = tracker.to_dict()
    assert rec1["metrics"]["psnr"]["best"] == 25.0
    assert rec1["metrics"]["psnr"]["epoch"] == 1
    assert rec1["metrics"]["ssim"]["best"] == 0.70
    assert rec1["metrics"]["ssim"]["epoch"] == 1
    assert rec1["metrics"]["lpips"]["best"] == 0.15
    assert rec1["metrics"]["lpips"]["epoch"] == 1

    # Epoch 2 validation: PSNR=24.5 (worse), SSIM=0.75 (better), LPIPS=0.18 (worse)
    tracker.update_validation(
        epoch=2,
        val_metrics={"val_psnr": 24.5, "val_ssim": 0.75, "val_lpips": 0.18},
    )
    rec2 = tracker.to_dict()
    # PSNR best remains 25.0 at epoch 1
    assert rec2["metrics"]["psnr"]["best"] == 25.0
    assert rec2["metrics"]["psnr"]["epoch"] == 1
    # SSIM best updates to 0.75 at epoch 2
    assert rec2["metrics"]["ssim"]["best"] == 0.75
    assert rec2["metrics"]["ssim"]["epoch"] == 2
    # LPIPS best remains 0.15 at epoch 1 (lower is better)
    assert rec2["metrics"]["lpips"]["best"] == 0.15
    assert rec2["metrics"]["lpips"]["epoch"] == 1

    # Epoch 3 validation: PSNR=29.41, SSIM=0.74, LPIPS=0.08 (better LPIPS!)
    tracker.update_validation(
        epoch=3,
        val_metrics={"val_psnr": 29.41, "val_ssim": 0.74, "val_lpips": 0.08},
    )
    rec3 = tracker.to_dict()
    assert rec3["metrics"]["psnr"]["best"] == 29.41
    assert rec3["metrics"]["psnr"]["epoch"] == 3
    assert rec3["metrics"]["lpips"]["best"] == 0.08
    assert rec3["metrics"]["lpips"]["epoch"] == 3

    # Verify YAML file was incrementally saved on disk
    record_file = tmp_path / "metrics_exp_record.yaml"
    with open(record_file, "r", encoding="utf-8") as f:
        saved_rec = yaml.safe_load(f)

    assert saved_rec["metrics"]["psnr"]["best"] == 29.41
    assert saved_rec["metrics"]["ssim"]["best"] == 0.75
    assert saved_rec["metrics"]["lpips"]["best"] == 0.08


def test_lpips_representation_when_unavailable(tmp_path):
    """Verify LPIPS is formatted cleanly as null/None when not evaluated or unavailable."""
    config = Config({"experiment_id": "no_lpips_exp"})
    tracker = ExperimentTracker(config=config, record_dir=tmp_path)

    # Validation without val_lpips
    tracker.update_validation(epoch=1, val_metrics={"val_psnr": 28.0, "val_ssim": 0.76})
    rec = tracker.to_dict()

    assert rec["metrics"]["psnr"]["best"] == 28.0
    assert rec["metrics"]["lpips"]["best"] is None
    assert rec["metrics"]["lpips"]["epoch"] is None

    # Inspect YAML content directly
    record_file = tmp_path / "no_lpips_exp_record.yaml"
    with open(record_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    assert yaml_data["metrics"]["lpips"]["best"] is None
    assert yaml_data["metrics"]["lpips"]["epoch"] is None
