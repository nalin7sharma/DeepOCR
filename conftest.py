"""Pytest configuration for DeepOCR tests."""
import sys
from pathlib import Path

# Ensure project root is on path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
