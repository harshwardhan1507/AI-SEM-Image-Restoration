"""Pytest unit test suite for utility modules (logger.py, seed.py)."""

import logging
from pathlib import Path

import numpy as np
import torch

from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def test_setup_logger_creates_file(tmp_path: Path) -> None:
    """Test setup_logger creates log directory and persistent log file."""
    log_dir = tmp_path / "test_logs"
    logger = setup_logger(name="TestLogger", log_dir=log_dir, log_level=logging.INFO)

    assert isinstance(logger, logging.Logger)
    assert log_dir.exists()
    log_files = list(log_dir.glob("execution_*.log"))
    assert len(log_files) == 1

    logger.info("Test log entry")
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "Test log entry" in log_content


def test_setup_logger_no_duplicate_handlers(tmp_path: Path) -> None:
    """Test calling setup_logger repeatedly does not duplicate log handlers."""
    log_dir = tmp_path / "dup_logs"
    logger1 = setup_logger(name="UniqueLogger", log_dir=log_dir)
    handler_count = len(logger1.handlers)

    logger2 = setup_logger(name="UniqueLogger", log_dir=log_dir)
    assert len(logger2.handlers) == handler_count


def test_set_seed_reproducibility() -> None:
    """Test set_seed produces identical NumPy and PyTorch random tensors."""
    set_seed(42, deterministic=True)
    np_rand1 = np.random.rand(5)
    torch_rand1 = torch.rand(5)

    set_seed(42, deterministic=True)
    np_rand2 = np.random.rand(5)
    torch_rand2 = torch.rand(5)

    assert np.allclose(np_rand1, np_rand2)
    assert torch.allclose(torch_rand1, torch_rand2)
