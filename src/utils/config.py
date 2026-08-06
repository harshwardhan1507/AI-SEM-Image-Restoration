"""YAML Configuration parser and schema validation module for SEM NAFNet restoration.

This module provides a hierarchical, immutable, dot-accessible configuration class
`Config` that parses default, training, model, and inference YAML configuration
files, merges experiment overrides, and handles environment variable resolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigDict(dict):
    """Dot-accessible recursive dictionary class for configuration management."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConfigDict with dictionary data.

        Args:
            data: Optional dictionary containing configuration keys and values.
        """
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set key and convert nested dictionaries into ConfigDict instances.

        Args:
            key: Configuration key name.
            value: Value associated with key.
        """
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)

    def __getattr__(self, item: str) -> Any:
        """Access dictionary items via dot-notation.

        Args:
            item: Key attribute name.

        Returns:
            Any: Value associated with key attribute.

        Raises:
            AttributeError: If key is not present in configuration.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{item}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        """Set key via attribute dot-notation.

        Args:
            key: Attribute key name.
            value: Value to assign.
        """
        self[key] = value

    def __delattr__(self, item: str) -> None:
        """Delete attribute key.

        Args:
            item: Attribute key name to remove.

        Raises:
            AttributeError: If key is not present in configuration.
        """
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{item}'") from None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict and all nested ConfigDict instances back to standard dict.

        Returns:
            Dict[str, Any]: Native Python dictionary representation.
        """
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def _recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dictionary d with keys from dictionary u.

    Args:
        d: Target dictionary to update.
        u: Source dictionary containing updates.

    Returns:
        Dict[str, Any]: Updated target dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _recursive_update(d[k], v)
        else:
            d[k] = v
    return d


class Config:
    """Master Configuration Loader and Schema Manager."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Config instance with optional base configuration dict.

        Args:
            config_dict: Base configuration dictionary.
        """
        self._data = ConfigDict(config_dict if config_dict else {})

    @property
    def data(self) -> ConfigDict:
        """Return raw ConfigDict object.

        Returns:
            ConfigDict: Dot-accessible configuration dictionary.
        """
        return self._data

    def __getattr__(self, item: str) -> Any:
        """Forward attribute lookup to underlying ConfigDict.

        Args:
            item: Configuration attribute name.

        Returns:
            Any: Value associated with configuration key.
        """
        return getattr(self._data, item)

    def __getitem__(self, item: str) -> Any:
        """Dictionary-style key lookup.

        Args:
            item: Key string.

        Returns:
            Any: Value associated with configuration key.
        """
        return self._data[item]

    def to_dict(self) -> Dict[str, Any]:
        """Return native dictionary representation of configuration.

        Returns:
            Dict[str, Any]: Nested dictionary representation.
        """
        return self._data.to_dict()

    def save_yaml(self, save_path: Union[str, Path]) -> None:
        """Save configuration instance to a YAML file.

        Args:
            save_path: Destination file path for YAML configuration output.
        """
        path = Path(save_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Construct Config instance from a single YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Config: Loaded Config instance.

        Raises:
            FileNotFoundError: If yaml_path does not exist on disk.
        """
        path = Path(yaml_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configuration YAML file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)

        return cls(content if content else {})

    @classmethod
    def load_combined(
        cls,
        config_files: List[Union[str, Path]],
        override_files: Optional[List[Union[str, Path]]] = None,
    ) -> "Config":
        """Load and merge multiple YAML configuration files sequentially.

        Args:
            config_files: Primary YAML configuration files to load in order.
            override_files: Optional list of experiment override YAML files.

        Returns:
            Config: Merged configuration object.
        """
        combined: Dict[str, Any] = {}

        for cfg_file in config_files:
            cfg_path = Path(cfg_file).resolve()
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        _recursive_update(combined, data)

        if override_files:
            for ovr_file in override_files:
                ovr_path = Path(ovr_file).resolve()
                if ovr_path.exists():
                    with open(ovr_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if data:
                            _recursive_update(combined, data)

        return cls(combined)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    experiment_path: Optional[Union[str, Path]] = None,
) -> Config:
    """Convenience helper function to load project configurations.

    Args:
        config_path: Path to primary training or model YAML file.
        experiment_path: Optional experiment override YAML path.

    Returns:
        Config: Loaded and merged configuration instance.
    """
    default_files = [
        Path("configs/default.yaml"),
        Path("configs/model.yaml"),
        Path("configs/train.yaml"),
    ]

    config_list: List[Union[str, Path]] = []
    for f in default_files:
        if f.exists():
            config_list.append(f)

    if config_path:
        config_list.append(config_path)

    override_list: Optional[List[Union[str, Path]]] = (
        [experiment_path] if experiment_path else None
    )

    return Config.load_combined(config_list, override_files=override_list)
