#!/usr/bin/env python
"""
Fine-tune the TrOCR Transformer (ViT encoder + Transformer decoder) on DeepOCR data.

Transfer learning: start from microsoft/trocr-small-handwritten and continue
training on a 50/50 mix of IAM handwriting and synthetic printed/script lines
(training partitions only).  Validation CER is measured on fixed IAM and
synthetic *validation* subsets before training and every ``--eval_every``
steps; the best weights are saved to ``checkpoints/trocr_finetuned/``.

Usage:
    python scripts/finetune_trocr.py --steps 300 --batch_size 8
    python scripts/evaluate.py --models trocr_ft
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.line_image import normalize_transcription
from evaluation.metrics import cer_score
from utils.config_loader import get_project_root
from utils.logger import get_logger

logger = get_logger(__name__)


def read_manifest(directory: Path, split: str) -> list[tuple[Path, str]]:
    rows = []
    for line in (directory / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            rows.append((directory / parts[0], normalize_transcription(parts[1])))
    return rows


def to_rgb(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if float(np.median(gray)) < 127:  # TrOCR expects dark ink on light paper
        gray = 255 - gray
    return np.stack([gray] * 3, axis=-1)


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(3, 12), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.5, 1.5), 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    return image


@torch.no_grad()
def evaluate(model, processor, rows, batch_size: int = 16) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        pixel_values = processor(images=[to_rgb(p) for p, _ in chunk], return_tensors="pt").pixel_values
        generated = model.generate(pixel_values, max_new_tokens=64, num_beams=1)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        total += sum(cer_score(normalize_transcription(t), ref) for t, (_, ref) in zip(texts, chunk))
    model.train()
    return 100 * total / max(1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eval_every", type=int, default=75)
    parser.add_argument("--eval_lines", type=int, default=80, help="Validation lines per dataset")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max_minutes", type=float, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

    root = get_project_root()
    rng = random.Random(42)
    torch.manual_seed(42)
    np.random.seed(42)
    iam_dir, synth_dir = root / "data" / "processed" / "iam", root / "data" / "processed" / "synthetic"
    train_iam, train_synth = read_manifest(iam_dir, "train"), read_manifest(synth_dir, "train")
    val = rng.sample(read_manifest(iam_dir, "val"), args.eval_lines) + rng.sample(read_manifest(synth_dir, "val"), args.eval_lines)
    val_iam, val_synth = val[:args.eval_lines], val[args.eval_lines:]

    processor = TrOCRProcessor.from_pretrained(args.base)
    model = VisionEncoderDecoderModel.from_pretrained(args.base)
    tokenizer = processor.tokenizer
    # TrOCR decodes from </s> (id 2, see its generation config) and its targets carry no
    # leading <s>; training with a different start token / label layout destroys the model.
    start_id = model.generation_config.decoder_start_token_id
    model.config.decoder_start_token_id = start_id if start_id is not None else tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, args.steps // 15), args.steps)
    out_dir = root / "checkpoints" / "trocr_finetuned"
    history_path = root / "results" / "training" / "trocr_ft_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    history = {"step": [], "train_loss": [], "val_cer": [], "val_cer_iam": [], "val_cer_synthetic": [], "val_loss": [], "train_cer": [], "val_wer": []}
    base_iam, base_synth = evaluate(model, processor, val_iam), evaluate(model, processor, val_synth)
    best = (base_iam + base_synth) / 2
    logger.info(f"Step 0 (pretrained): val CER IAM={base_iam:.2f}% synthetic={base_synth:.2f}% mean={best:.2f}%")
    for key, value in (("step", 0), ("train_loss", None), ("val_cer", best / 100), ("val_cer_iam", base_iam), ("val_cer_synthetic", base_synth)):
        history[key].append(value)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    running = []
    for step in range(1, args.steps + 1):
        batch = [rng.choice(train_iam) if i % 2 == 0 else rng.choice(train_synth) for i in range(args.batch_size)]
        pixel_values = processor(images=[augment(to_rgb(p), rng) for p, _ in batch], return_tensors="pt").pixel_values
        labels = tokenizer([text for _, text in batch], padding="longest", truncation=True, max_length=64, return_tensors="pt").input_ids
        if bool((labels[:, 0] == tokenizer.bos_token_id).all()):
            labels = labels[:, 1:]  # drop <s>; the decoder start token plays that role
        labels = labels.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running.append(loss.item())

        if step % 10 == 0:
            logger.info(f"step {step}/{args.steps} loss={np.mean(running[-10:]):.4f} ({(time.time() - started) / 60:.1f} min)")
        last = step == args.steps or (args.max_minutes and (time.time() - started) / 60 > args.max_minutes)
        if step % args.eval_every == 0 or last:
            cer_iam, cer_synth = evaluate(model, processor, val_iam), evaluate(model, processor, val_synth)
            mean = (cer_iam + cer_synth) / 2
            logger.info(f"step {step}: val CER IAM={cer_iam:.2f}% synthetic={cer_synth:.2f}% mean={mean:.2f}% (best {best:.2f}%)")
            for key, value in (("step", step), ("train_loss", float(np.mean(running[-args.eval_every:]))),
                               ("val_cer", mean / 100), ("val_cer_iam", cer_iam), ("val_cer_synthetic", cer_synth)):
                history[key].append(value)
            if mean < best:
                best = mean
                model.save_pretrained(out_dir)
                processor.save_pretrained(out_dir)
                logger.info(f"Saved improved weights to {out_dir}")
            history_path.write_text(json.dumps({
                "model": "trocr_ft", "epoch": len(history["step"]) - 1, "epochs_planned": args.steps // args.eval_every,
                "finished": bool(last), "best_val_cer": best / 100, "epoch_seconds": [],
                "elapsed_seconds": round(time.time() - started, 1),
                "meta": {"base": args.base, "steps": step, "batch_size": args.batch_size, "lr": args.lr,
                         "parameters": sum(p.numel() for p in model.parameters()), "datasets": ["iam", "synthetic"]},
                "history": history,
            }, indent=2), encoding="utf-8")
        if last:
            break
    logger.info(f"Done. Best mean val CER {best:.2f}% — weights in {out_dir}")


if __name__ == "__main__":
    main()
