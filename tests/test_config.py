"""Pytest unit test suite for configuration system (src/utils/config.py)."""

from pathlib import Path

import pytest

from src.utils.config import Config, ConfigDict, load_config


def test_config_dict_dot_access() -> None:
    """Test ConfigDict initialization and dot-notation attribute access."""
    data = {"model": {"width": 32, "name": "NAFNet"}, "train": {"batch_size": 16}}
    cfg = ConfigDict(data)

    assert cfg.model.width == 32
    assert cfg.model.name == "NAFNet"
    assert cfg.train.batch_size == 16


def test_config_dict_attribute_error() -> None:
    """Test AttributeError raised for missing configuration keys."""
    cfg = ConfigDict({"key": "val"})
    with pytest.raises(AttributeError):
        _ = cfg.non_existent_key


def test_config_dict_del_attribute() -> None:
    """Test deleting attribute key via delattr."""
    cfg = ConfigDict({"key": "val"})
    del cfg.key
    assert "key" not in cfg

    with pytest.raises(AttributeError):
        del cfg.non_existent_key


def test_config_dict_to_dict() -> None:
    """Test converting ConfigDict and nested ConfigDict instances back to native dict."""
    data = {"dataset": {"patch_size": 128}, "seed": 42}
    cfg = ConfigDict(data)
    d = cfg.to_dict()

    assert isinstance(d, dict)
    assert not isinstance(d["dataset"], ConfigDict)
    assert d["dataset"]["patch_size"] == 128
    assert d["seed"] == 42


def test_config_from_yaml(tmp_path: Path) -> None:
    """Test loading configuration from a YAML file."""
    yaml_content = "train:\n  batch_size: 32\n  learning_rate: 0.001\n"
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = Config.from_yaml(yaml_file)
    assert cfg.train.batch_size == 32
    assert cfg.train.learning_rate == 0.001


def test_config_from_yaml_not_found() -> None:
    """Test FileNotFoundError raised when YAML path does not exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("non_existent_path_12345.yaml")


def test_config_load_combined(tmp_path: Path) -> None:
    """Test loading and recursively merging primary and override YAML configs."""
    base_content = "model:\n  width: 32\n  depth: 4\ntrain:\n  batch_size: 16\n"
    override_content = "model:\n  width: 64\n"

    base_file = tmp_path / "base.yaml"
    override_file = tmp_path / "override.yaml"
    base_file.write_text(base_content, encoding="utf-8")
    override_file.write_text(override_content, encoding="utf-8")

    cfg = Config.load_combined([base_file], override_files=[override_file])
    assert cfg.model.width == 64
    assert cfg.model.depth == 4
    assert cfg.train.batch_size == 16


def test_config_save_yaml(tmp_path: Path) -> None:
    """Test saving configuration object back to disk as YAML."""
    data = {"experiment": "exp001", "params": {"lr": 0.0005}}
    cfg = Config(data)
    save_file = tmp_path / "saved_config.yaml"
    cfg.save_yaml(save_file)

    assert save_file.exists()
    loaded_cfg = Config.from_yaml(save_file)
    assert loaded_cfg.experiment == "exp001"
    assert loaded_cfg.params.lr == 0.0005


def test_load_config_helper(tmp_path: Path) -> None:
    """Test load_config helper function with custom config path."""
    cfg_file = tmp_path / "custom_train.yaml"
    cfg_file.write_text("train:\n  epochs: 50\n", encoding="utf-8")

    cfg = load_config(config_path=cfg_file)
    assert cfg.train.epochs == 50
