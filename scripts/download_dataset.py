#!/usr/bin/env python
"""
Dataset Download Helper for DeepOCR.

IAM Handwriting Database:
  Registration required at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
  This script provides instructions and validates the download once complete.

For development without IAM, use the synthetic dataset:
  python scripts/prepare_dataset.py --dataset synthetic --n_samples 2000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import get_project_root
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    root = get_project_root()
    iam_dir = root / "data" / "raw" / "IAM"

    print("""
=======================================================================
IAM Handwriting Database Download Instructions
=======================================================================

The IAM dataset requires free registration. Follow these steps:

1. Register at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

2. Download the following files:
   - lines.tgz       (line-level segmented images)
   - ascii.tgz       (ground-truth transcriptions)

3. Extract them into:
""")
    print(f"   {iam_dir}")
    print("""
   Expected structure after extraction:
   data/raw/IAM/
   ├── lines/
   │   ├── a01/
   │   │   ├── a01-000u/
   │   │   │   ├── a01-000u-00.png
   │   │   │   └── ...
   │   └── ...
   └── ascii/
       └── lines.txt

4. After extraction, run:
   python scripts/prepare_dataset.py --dataset iam

=======================================================================
Alternative: Synthetic Dataset (no registration needed)
=======================================================================
For quick development without IAM, generate a synthetic dataset:
   python scripts/prepare_dataset.py --dataset synthetic --n_samples 2000

Then train with:
   python scripts/train.py --model cnn_bilstm

=======================================================================
""")


if __name__ == "__main__":
    main()
