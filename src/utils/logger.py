"""Centralized logging infrastructure module for SEM NAFNet restoration.

This module provides a dual-channel logging utility `setup_logger` that formats
execution logs for simultaneous output to console (sys.stdout) and persistent
disk log files inside the configured logging directory.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "SEM_NAFNet",
    log_dir: Optional[Union[str, Path]] = "logs",
    log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
) -> logging.Logger:
    """Initialize and configure a dual console/file logging instance.

    Args:
        name: Name of logger instance.
        log_dir: Directory path to store execution log files. If None, file logging is disabled.
        log_level: Console logging output verbosity level.
        file_log_level: File logging output verbosity level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"execution_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
