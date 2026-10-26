#!/usr/bin/env python
"""
Evaluation Script for DeepOCR models on the held-out TEST partitions.

Models:
  cnn_bilstm, crnn  trained here (checkpoints/<model>/best_model.pth required —
                    the script refuses to score untrained/random weights)
  trocr             pretrained Transformer OCR (transfer learning, no training here)
  paddle            PaddleOCR pretrained recogniser (industrial reference)

Test sets: ``iam`` (official writer-independent IAM test lines) and
``synthetic`` (printed + handwriting-font lines with degradations).

Metrics per (model, set): CER, WER, exact-line accuracy (all in %, averaged
over lines, on normalised transcriptions — see datasets/line_image.py), raw
CER without punctuation-spacing normalisation, and mean inference ms/line.

Outputs (merged across runs, so models can be evaluated separately):
  results/evaluation/summary.json          read by the web dashboard
  results/evaluation/samples.json          fixed random test lines + predictions
  results/evaluation/predictions/*.csv     every prediction
  results/model_comparison.csv / .md       tables for the report
  results/plots/model_comparison.png       bar chart for the report

Usage:
    python scripts/evaluate.py --all
    python scripts/evaluate.py --models crnn cnn_bilstm
    python scripts/evaluate.py --models trocr --limit 500 --threads 4
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
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
from evaluation.metrics import cer_score, wer_score
from utils.config_loader import get_project_root, load_all_configs
from utils.logger import get_logger

logger = get_logger(__name__)

ALL_MODELS = ["cnn_bilstm", "crnn", "trocr", "trocr_ft", "paddle"]
SET_LABELS = {
    "iam": ("IAM handwriting", "Official IAM writer-independent test lines (unseen writers)"),
    "synthetic": ("Synthetic printed + script", "Rendered lines with 32 fonts and realistic degradations"),
}
MODEL_INFO = {
    "cnn_bilstm": {"label": "CNN + BiLSTM (baseline)", "architecture": "CNN-BiLSTM-CTC", "type": "trained from scratch"},
    "crnn": {"label": "CRNN", "architecture": "CRNN (VGG + BiLSTM + CTC)", "type": "trained from scratch"},
    "trocr": {"label": "TrOCR Transformer (zero-shot)", "architecture": "ViT encoder + Transformer decoder", "type": "pretrained / transfer learning"},
    "trocr_ft": {"label": "TrOCR Transformer (fine-tuned)", "architecture": "ViT encoder + Transformer decoder", "type": "pretrained + fine-tuned on our data"},
    "paddle": {"label": "PaddleOCR", "architecture": "PP-OCR recogniser", "type": "pretrained reference"},
}


def require_checkpoint(checkpoint_path: Path) -> Path:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}. "
            "Train the model first; refusing to report metrics for random weights."
        )
    return checkpoint_path


def load_test_set(root: Path, cfg: dict, name: str, limit: int | None) -> list[tuple[Path, str]]:
    section = cfg.get(name) or {}
    directory = root / section.get("processed_dir", f"data/processed/{name}")
    manifest = directory / "test.txt"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing test manifest {manifest}. Prepare the dataset first.")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            rows.append((directory / parts[0], parts[1].strip()))
    if limit and len(rows) > limit:
        rows = random.Random(42).sample(rows, limit)  # fixed, reproducible subset
        rows.sort(key=lambda item: str(item[0]))
    return rows


def build_recognizer(name: str, root: Path, cfg: dict):
    from models.recognizer import LineRecognizer, PaddleLineRecognizer, TrOCRRecognizer

    if name in ("cnn_bilstm", "crnn"):
        path = require_checkpoint(root / "checkpoints" / name / "best_model.pth")
        return LineRecognizer.from_checkpoint(path)
    if name == "trocr":
        return TrOCRRecognizer(cfg.get("transformer_ocr", {}).get("encoder", "microsoft/trocr-small-handwritten"))
    if name == "trocr_ft":
        path = root / "checkpoints" / "trocr_finetuned"
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"No fine-tuned TrOCR at {path}. Run: python scripts/finetune_trocr.py")
        return TrOCRRecognizer(str(path))
    if name == "paddle":
        return PaddleLineRecognizer()
    raise ValueError(name)


def score(predictions: list[str], references: list[str]) -> dict:
    norm_pred = [normalize_transcription(p) for p in predictions]
    norm_ref = [normalize_transcription(r) for r in references]
    n = max(1, len(references))
    raw_ref = [" ".join(r.split()) for r in references]
    raw_pred = [" ".join(p.split()) for p in predictions]
    return {
        "cer": round(100 * sum(cer_score(p, r) for p, r in zip(norm_pred, norm_ref)) / n, 3),
        "wer": round(100 * sum(wer_score(p, r) for p, r in zip(norm_pred, norm_ref)) / n, 3),
        "accuracy": round(100 * sum(p == r for p, r in zip(norm_pred, norm_ref)) / n, 3),
        "cer_raw": round(100 * sum(cer_score(p, r) for p, r in zip(raw_pred, raw_ref)) / n, 3),
        "samples": len(references),
    }


def encode_png(path: Path, max_width: int = 900) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ""
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".png", image)
    return base64.b64encode(data.tobytes()).decode("ascii") if ok else ""


def _fmt(value, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def write_report_tables(summary: dict, results_dir: Path) -> None:
    sets = list(summary.get("test_sets", {}))
    rows = []
    for model in [key for base in ALL_MODELS for key in (base, f"{base}_pp")]:
        entry = summary["models"].get(model)
        if not entry:
            continue
        info = entry.get("_info") or MODEL_INFO[model.removesuffix("_pp")]
        row = {"model": info.get("label", model), "architecture": info.get("architecture", "")}
        speeds = []
        for set_name in sets:
            metrics = entry.get(set_name)
            if metrics:
                row[f"{set_name}_cer"] = metrics["cer"]
                row[f"{set_name}_wer"] = metrics["wer"]
                row[f"{set_name}_accuracy"] = metrics["accuracy"]
                speeds.append(metrics["ms_per_line"])
        row["ms_per_line"] = round(sum(speeds) / len(speeds), 2) if speeds else ""
        rows.append(row)
    if not rows:
        return
    fields = ["model", "architecture"] + [f"{s}_{m}" for s in sets for m in ("cer", "wer", "accuracy")] + ["ms_per_line"]
    with open(results_dir / "model_comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    header = "| Model | Architecture | " + " | ".join(f"{SET_LABELS.get(s, (s,))[0]} CER | WER | Acc" for s in sets) + " | ms/line |"
    divider = "|" + "---|" * (3 + 3 * len(sets))
    lines = [header, divider]
    for row in rows:
        cells = [row["model"], row["architecture"]]
        for s in sets:
            cells += [_fmt(row.get(f"{s}_cer")), _fmt(row.get(f"{s}_wer")), _fmt(row.get(f"{s}_accuracy"))]
        cells.append(_fmt(row["ms_per_line"], 1))
        lines.append("| " + " | ".join(cells) + " |")
    (results_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_readme(results_dir.parent / "README.md", lines, summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 4.2), squeeze=False)
        palette = {"cnn_bilstm": "#06b6d4", "crnn": "#8b5cf6", "trocr": "#65a30d", "trocr_ft": "#ca8a04", "paddle": "#64748b"}
        for ax, set_name in zip(axes[0], sets):
            models = [m for m in ALL_MODELS if summary["models"].get(m, {}).get(set_name)]
            cers = [summary["models"][m][set_name]["cer"] for m in models]
            wers = [summary["models"][m][set_name]["wer"] for m in models]
            x = np.arange(len(models))
            ax.bar(x - 0.2, cers, 0.4, label="CER %", color=[palette[m] for m in models])
            ax.bar(x + 0.2, wers, 0.4, label="WER %", color=[palette[m] for m in models], alpha=0.45)
            ax.set_xticks(x, [MODEL_INFO[m]["label"].split(" (")[0] for m in models], rotation=12)
            ax.set_title(SET_LABELS.get(set_name, (set_name,))[0])
            ax.set_ylabel("Error rate (%) — lower is better")
            for xi, value in zip(x, cers):
                ax.text(xi - 0.2, value + 0.5, f"{value:.1f}", ha="center", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        axes[0][0].legend()
        fig.tight_layout()
        (results_dir / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "plots" / "model_comparison.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best effort
        logger.warning(f"Could not draw comparison plot: {exc}")


def update_readme(readme: Path, table_lines: list[str], summary: dict) -> None:
    """Replace the README's results block (between RESULTS markers) with measured numbers."""
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError:
        return
    if start not in text or end not in text:
        return
    sets = summary.get("test_sets", {})
    described = "; ".join(f"**{v['label']}**: {v['samples']} held-out test lines" for v in sets.values())
    block = "\n".join([
        start,
        "## Results",
        "",
        f"Measured with `scripts/evaluate.py` ({summary.get('generated_at', '')}) on {described}.",
        "CER, WER and accuracy are percentages averaged per line (lower CER/WER is better). ms/line is CPU time.",
        "",
        *table_lines,
        "",
        *[f"- {note}" for note in summary.get("notes", [])],
        "",
        "Charts: `results/plots/model_comparison.png`, `results/plots/robustness.png` and",
        "`results/plots/*_training_curves.png`. All of them are interactive on the web app's **Research** page.",
        end,
    ])
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    readme.write_text(head + block + tail, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepOCR models on held-out test sets")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=None)
    parser.add_argument("--all", action="store_true", help="Evaluate every available model")
    parser.add_argument("--sets", nargs="+", default=["iam", "synthetic"])
    parser.add_argument("--limit", type=int, default=None, help="Fixed random subset size per test set")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--samples", type=int, default=8, help="Sample predictions saved per set")
    parser.add_argument("--postprocess", action="store_true",
                        help="Also apply post-processing (TextCorrector) and store results as <model>_pp")
    # Backwards-compatible single-model flags
    parser.add_argument("--model", choices=ALL_MODELS, default=None)
    parser.add_argument("--checkpoint", default=None, help="(legacy) checkpoint for --model")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_all_configs()
    root = get_project_root()
    results_dir = root / "results"
    eval_dir = results_dir / "evaluation"
    (eval_dir / "predictions").mkdir(parents=True, exist_ok=True)
    models = args.models or ([args.model] if args.model else None) or (ALL_MODELS if args.all else ["cnn_bilstm", "crnn"])

    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"models": {}}
    summary.setdefault("models", {})
    summary.setdefault("test_sets", {})
    samples_path = eval_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.exists() else []

    test_sets = {name: load_test_set(root, cfg, name, args.limit) for name in args.sets}
    for name, rows in test_sets.items():
        label, description = SET_LABELS.get(name, (name, ""))
        summary["test_sets"][name] = {"label": label, "description": description, "samples": len(rows)}

    # Fixed sample lines shown in the dashboard (same lines for every model).
    sample_keys = {name: random.Random(7).sample(range(len(rows)), min(args.samples, len(rows))) for name, rows in test_sets.items()}
    sample_index = {(s["set"], s["image_path"]): s for s in samples if "image_path" in s}

    for model_name in models:
        try:
            if args.model and args.checkpoint and model_name == args.model and model_name in ("cnn_bilstm", "crnn"):
                from models.recognizer import LineRecognizer
                recognizer = LineRecognizer.from_checkpoint(require_checkpoint(Path(args.checkpoint)))
            else:
                recognizer = build_recognizer(model_name, root, cfg)
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error(f"Skipping {model_name}: {exc}")
            continue
        corrector = None
        result_key = model_name
        if args.postprocess:
            from postprocessing.text_corrector import TextCorrector

            corrector = TextCorrector()
            result_key = f"{model_name}_pp"
        entry = summary["models"].setdefault(result_key, {})
        info = dict(MODEL_INFO[model_name])
        if corrector is not None:
            info["label"] = f"{info['label']} + post-processing"
            info["type"] = f"{info['type']}, with error correction"
        if hasattr(recognizer, "describe"):
            info.update({k: v for k, v in recognizer.describe().items()
                         if v is not None and k not in ("architecture", "label")})
        entry["_info"] = info

        for set_name, rows in test_sets.items():
            images = [cv2.imread(str(path)) for path, _ in rows]
            refs = [text for _, text in rows]
            logger.info(f"{model_name} on {set_name}: {len(rows)} lines ...")
            started = time.perf_counter()
            outputs = recognizer.recognize(images)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if corrector is not None:
                outputs = [(corrector.correct(text), conf) for text, conf in outputs]
            preds = [text for text, _ in outputs]
            metrics = score(preds, refs)
            metrics["ms_per_line"] = round(elapsed_ms / max(1, len(rows)), 2)
            metrics["mean_confidence"] = round(float(np.mean([c for _, c in outputs])) if outputs else 0.0, 4)
            entry[set_name] = metrics
            logger.info(f"  {result_key}/{set_name}: CER={metrics['cer']:.2f}% WER={metrics['wer']:.2f}% "
                        f"Acc={metrics['accuracy']:.2f}% ({metrics['ms_per_line']:.1f} ms/line)")

            with open(eval_dir / "predictions" / f"{result_key}_{set_name}.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_path", "ground_truth", "prediction", "confidence", "cer"])
                for (path, ref), (pred, conf) in zip(rows, outputs):
                    writer.writerow([path.relative_to(root).as_posix(), ref, pred, round(conf, 4),
                                     round(cer_score(normalize_transcription(pred), normalize_transcription(ref)), 4)])

            for index in sample_keys[set_name]:
                path, ref = rows[index]
                key = (set_name, path.relative_to(root).as_posix())
                if key not in sample_index:
                    sample_index[key] = {"set": set_name, "image_path": key[1], "image": encode_png(path),
                                         "ground_truth": normalize_transcription(ref), "predictions": {}}
                pred = normalize_transcription(outputs[index][0])
                sample_index[key]["predictions"][result_key] = {
                    "text": pred, "cer": round(cer_score(pred, normalize_transcription(ref)), 4)}

        summary["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        summary["notes"] = [
            "CER/WER/accuracy are averaged per line on the held-out test partitions; lower CER/WER is better.",
            "Transcriptions are normalised (IAM's tokenised spacing before punctuation is removed) for every model alike; cer_raw keeps the original spacing.",
            "CNN-BiLSTM and CRNN were trained from scratch on CPU (IAM train + synthetic train only); TrOCR and PaddleOCR are pretrained by their authors.",
            "TrOCR-small-handwritten was fine-tuned by Microsoft on IAM, so its IAM score may be optimistic if its training partition overlaps this test split.",
        ]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        samples_path.write_text(json.dumps(list(sample_index.values()), indent=1), encoding="utf-8")
        write_report_tables(summary, results_dir)
        del recognizer

    print("\n" + json.dumps({m: {s: summary["models"][m].get(s) for s in args.sets} for m in models if m in summary["models"]}, indent=2))


if __name__ == "__main__":
    main()
