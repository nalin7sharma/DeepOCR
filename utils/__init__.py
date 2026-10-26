"""Utility modules for DeepOCR."""
from .logger import get_logger
from .checkpoint import save_checkpoint, load_checkpoint
from .config_loader import load_config, get_project_root

__all__ = ["get_logger", "save_checkpoint", "load_checkpoint",
           "load_config", "get_project_root"]
