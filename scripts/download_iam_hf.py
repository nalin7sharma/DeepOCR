#!/usr/bin/env python
"""
Download the IAM handwriting *line* dataset from the Hugging Face Hub and
export it to the simple DeepOCR layout used by ``scripts/train.py``.

Source: https://huggingface.co/datasets/Teklia/IAM-line (MIT-licensed export,
official train / validation / test partitions, lines resized to 128 px height).

Output layout::

    data/processed/iam/
        images/train/000001.png ...
        images/val/...
        images/test/...
        train.txt   (relative_image_path<TAB>transcription)
        val.txt
        test.txt
        charset.txt (every character that occurs in the transcriptions)

Usage:
    python scripts/download_iam_hf.py
    python scripts/download_iam_hf.py --max_train 3000   # smaller subset
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "Teklia/IAM-line"
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "processed" / "iam"))
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap on training lines")
    parser.add_argument("--max_eval", type=int, default=None, help="Optional cap on val/test lines")
    args = parser.parse_args()

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(f"Missing dependency: {exc}. Install pandas, pyarrow and huggingface_hub.")

    out_dir = Path(args.out_dir)
    charset: set[str] = set()

    for hf_split, split in SPLITS.items():
        print(f"[IAM] downloading {hf_split} parquet ...", flush=True)
        parquet_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision="refs/convert/parquet",
            filename=f"default/{hf_split}/0000.parquet",
        )
        frame = pd.read_parquet(parquet_path)
        limit = args.max_train if split == "train" else args.max_eval
        if limit:
            frame = frame.iloc[:limit]

        image_dir = out_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        for index, record in enumerate(frame.itertuples(index=False), start=1):
            text = " ".join(str(record.text).split())
            if not text:
                continue
            image_field = record.image
            payload = image_field["bytes"] if isinstance(image_field, dict) else image_field
            relative = f"images/{split}/{index:06d}.png"
            target = out_dir / relative
            if not target.exists():
                Image.open(io.BytesIO(payload)).convert("L").save(target)
            rows.append(f"{relative}\t{text}")
            charset.update(text)
        (out_dir / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"[IAM] {split}: {len(rows)} lines -> {out_dir / f'{split}.txt'}", flush=True)

    (out_dir / "charset.txt").write_text("".join(sorted(charset)), encoding="utf-8")
    print(f"[IAM] charset ({len(charset)} chars): {''.join(sorted(charset))}")


if __name__ == "__main__":
    main()
