"""Main entry point script for training NAFNet on SEM images.

This script parses command line arguments, loads configuration files, initializes
seed and logging infrastructure, builds data loaders, NAFNet model, loss function,
optimizer, learning rate scheduler, checkpoint manager, and TensorBoard writer,
and executes the training loop via Trainer.fit().

Example:
    Run training with default configuration:
        $ python train.py --config configs/train.yaml

    Run training with specific experiment configuration:
        $ python train.py --config configs/train.yaml --experiment configs/experiments/exp001.yaml

    Resume training from a checkpoint:
        $ python train.py --config configs/train.yaml --resume outputs/checkpoints/checkpoint_epoch_010.pth
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.datasets.builder import build_dataloaders
from src.engine.checkpoint import CheckpointManager
from src.engine.trainer import Trainer
from src.losses.builder import build_loss
from src.models.builder import build_model
from src.utils.config import Config, load_config
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for training entry point.

    Args:
        args: Optional list of argument strings to parse. If None, uses sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train NAFNet model for SEM image restoration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to primary training configuration YAML file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Path to experiment configuration override YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint .pth file to resume training from.",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="Path to model checkpoint .pth file to finetune from (loads weights only).",
    )
    return parser.parse_args(args)


def resolve_device(device_setting: str) -> str:
    """Resolve device setting to explicit 'cuda' or 'cpu' string.

    Args:
        device_setting: String device setting (e.g. 'auto', 'cuda', 'cpu').

    Returns:
        str: Resolved device string ('cuda' or 'cpu').
    """
    if device_setting.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_setting


def _get_config_val(config: Config, key_paths: List[str], default: Any = None) -> Any:
    """Extract configuration value trying multiple candidate dot-delimited key paths.

    Args:
        config: Config instance to extract values from.
        key_paths: List of dot-delimited key path strings to try.
        default: Fallback default value if no key path matches.

    Returns:
        Any: Extracted value or default.
    """
    cfg_dict = config.to_dict()
    for key_path in key_paths:
        parts = key_path.split(".")
        curr: Any = cfg_dict
        found = True
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                found = False
                break
        if found and curr is not None:
            return curr
    return default


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Main training entry point execution function.

    Args:
        args: Optional pre-parsed CLI arguments Namespace. If None, parses sys.argv.

    Returns:
        Dict[str, Any]: Training summary dictionary from Trainer.fit().

    Raises:
        FileNotFoundError: If requested config, experiment, or resume files do not exist.
        ValueError: If configuration or dataloader construction fails.
    """
    if args is None:
        args = parse_args()

    # 1. Validate configuration & checkpoint paths existence
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if args.experiment:
        exp_path = Path(args.experiment).resolve()
        if not exp_path.exists():
            raise FileNotFoundError(
                f"Experiment configuration override file not found: {exp_path}"
            )

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume_path}")

    if args.finetune:
        finetune_path = Path(args.finetune).resolve()
        if not finetune_path.exists():
            raise FileNotFoundError(f"Finetune checkpoint file not found: {finetune_path}")

    # 2. Load configuration using existing Config infrastructure
    config = load_config(config_path=args.config, experiment_path=args.experiment)

    # 3. Initialize random seed
    seed = int(
        _get_config_val(config, ["system.seed", "train.seed", "seed"], default=42)
    )
    set_seed(seed)

    # 4. Initialize logger
    log_dir = _get_config_val(
        config, ["system.log_dir", "train.log_dir", "log_dir"], default="logs"
    )
    logger = setup_logger(name="SEM_NAFNet_Train", log_dir=log_dir)
    logger.info(f"Loaded configuration from {config_path}")
    if args.experiment:
        logger.info(f"Applied experiment override from {args.experiment}")

    # 5. Resolve device string
    device_setting = str(
        _get_config_val(
            config, ["system.device", "train.device", "device"], default="auto"
        )
    )
    device_str = resolve_device(device_setting)
    logger.info(f"Resolved execution device: {device_str}")

    # 6. Build DataLoaders
    env_dataset_dir = os.environ.get("SEM_DATASET_ROOT") or os.environ.get(
        "SEM_DATASET_DIR"
    )
    dataset_dir = env_dataset_dir or _get_config_val(
        config, ["data.dataset_dir", "dataset_dir", "data.dataset_path"], default=None
    )
    loaders = build_dataloaders(config=config, dataset_dir=dataset_dir)
    logger.info(f"DataLoaders built successfully for splits: {list(loaders.keys())}")

    # 7. Build NAFNet Model
    model = build_model(config)
    model = model.to(device_str)
    logger.info(f"Built model: {type(model).__name__}")

    # 8. Build Loss Function
    criterion = build_loss(config)
    logger.info(f"Built loss function: {type(criterion).__name__}")

    # 9. Extract Optimization Hyperparameters & Construct AdamW & CosineAnnealingLR
    epochs = int(_get_config_val(config, ["train.epochs", "epochs"], default=100))
    lr = float(
        _get_config_val(
            config,
            ["train.learning_rate", "train.lr", "learning_rate", "lr"],
            default=1e-3,
        )
    )
    weight_decay = float(
        _get_config_val(config, ["train.weight_decay", "weight_decay"], default=1e-4)
    )
    min_lr = float(
        _get_config_val(
            config, ["train.min_lr", "train.eta_min", "min_lr"], default=1e-6
        )
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # 10. Construct CheckpointManager
    checkpoint_dir = _get_config_val(
        config,
        ["system.checkpoint_dir", "train.checkpoint_dir", "checkpoint_dir"],
        default="outputs/checkpoints",
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    # 11. Construct SummaryWriter
    tensorboard_dir = _get_config_val(
        config,
        ["system.tensorboard_dir", "train.tensorboard_dir", "tensorboard_dir"],
        default="outputs/tensorboard",
    )
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # 12. Handle --resume or --finetune
    start_epoch = 1
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        logger.info(f"Loading resume checkpoint from {checkpoint_path}")
        checkpoint = checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device_str,
        )
        if checkpoint and "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming training starting at epoch {start_epoch}")
    elif args.finetune:
        finetune_path = Path(args.finetune).resolve()
        logger.info(f"Loading finetune weights from {finetune_path}")
        # Only load the model weights, leave optimizer/scheduler fresh
        checkpoint_manager.load(
            checkpoint_path=finetune_path,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=device_str,
        )
        logger.info(f"Successfully loaded finetune weights.")

    # 13. Construct ExperimentTracker & Trainer
    metrics_config = _get_config_val(
        config, ["metrics"], default={"psnr": True, "ssim": True, "lpips": False}
    )
    experiment_tracker = ExperimentTracker(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataset_dir=dataset_dir,
    )
    logger.info(f"Initialized ExperimentTracker record: {experiment_tracker.record_file_path}")

    use_amp = bool(
        _get_config_val(
            config, ["train.mixed_precision", "train.use_amp", "use_amp"], default=False
        )
    )
    amp_dtype = str(
        _get_config_val(config, ["train.amp_dtype", "amp_dtype"], default="float16")
    )
    grad_clip_norm_val = _get_config_val(
        config,
        ["train.gradient_clip_val", "train.grad_clip_norm", "grad_clip_norm"],
        default=None,
    )
    grad_clip_norm = (
        float(grad_clip_norm_val) if grad_clip_norm_val is not None else None
    )
    val_freq = int(_get_config_val(config, ["train.val_freq", "val_freq"], default=1))
    log_freq = int(_get_config_val(config, ["train.log_freq", "log_freq"], default=10))

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        val_loader=loaders.get("val") or loaders.get("test"),
        checkpoint_manager=checkpoint_manager,
        writer=writer,
        device=device_str,
        epochs=epochs,
        grad_clip_norm=grad_clip_norm,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        val_freq=val_freq,
        log_freq=log_freq,
        experiment_tracker=experiment_tracker,
        metrics_config=metrics_config,
    )

    # 14. Start training loop
    logger.info(
        f"Starting training run (Epochs {start_epoch} to {epochs}, Device: {device_str})..."
    )
    summary = trainer.fit(start_epoch=start_epoch)
    summary["experiment_record_path"] = str(experiment_tracker.record_file_path)
    logger.info(
        f"Training completed successfully. Total epochs completed: {summary['epochs_completed']}"
    )

    return summary


if __name__ == "__main__":
    main()
