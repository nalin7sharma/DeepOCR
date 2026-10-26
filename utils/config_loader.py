"""Configuration loading utilities."""
import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file lives at <root>/utils/config_loader.py
    return Path(__file__).resolve().parent.parent


def load_config(config_name: str = "default") -> dict:
    """Load a YAML config file from the configs/ directory.

    Args:
        config_name: Name without extension (e.g. "default", "training").

    Returns:
        Parsed config as a nested dict.
    """
    root = get_project_root()
    config_path = root / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_all_configs() -> dict:
    """Load and merge all standard config files into one dict."""
    cfg = {}
    for name in ("default", "dataset", "model", "training"):
        try:
            cfg.update(load_config(name))
        except FileNotFoundError:
            pass
    return cfg
