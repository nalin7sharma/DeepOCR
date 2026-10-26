"""
Model Evaluator for DeepOCR.

Runs inference on a test DataLoader, collects predictions, computes
CER/WER/accuracy/inference time and saves results to disk.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.ctc_utils import ctc_greedy_decode
from .metrics import cer_score, wer_score, recognition_accuracy, batch_cer, batch_wer
from utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluate an OCR model on a DataLoader.

    Args:
        model: Trained OCR model.
        test_loader: DataLoader providing test samples.
        vocab: (chars, char2idx, idx2char) vocabulary tuple.
        model_type: "ctc" | "seq2seq"
        device: Torch device.
        results_dir: Directory to save evaluation outputs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        vocab: tuple,
        model_type: str = "ctc",
        device: torch.device | str = "auto",
        results_dir: str | Path = "results",
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.vocab = vocab
        self.model_type = model_type
        self.results_dir = Path(results_dir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, model_name: str = "model") -> dict:
        """Run full evaluation and save results.

        Args:
            model_name: Label used in saved files.

        Returns:
            Dict with keys: cer, wer, accuracy, inference_ms, num_samples,
            predictions (list of dicts with pred/gt/cer/wer).
        """
        self.model.eval()
        all_preds: list[str] = []
        all_gts: list[str] = []
        total_time = 0.0
        total_samples = 0

        for batch in self.test_loader:
            images = batch["image"].to(self.device)
            label_texts = batch["label_text"]
            B = images.size(0)

            t0 = time.perf_counter()
            if self.model_type == "ctc":
                log_probs = self.model(images)
                preds = ctc_greedy_decode(log_probs.cpu(), self.vocab[2])
            else:
                # Greedy seq2seq
                preds = self._seq2seq_infer(images)
            elapsed = time.perf_counter() - t0

            total_time += elapsed
            total_samples += B
            all_preds.extend(preds)
            all_gts.extend(label_texts)

        avg_cer = batch_cer(all_preds, all_gts) * 100
        avg_wer = batch_wer(all_preds, all_gts) * 100
        acc = recognition_accuracy(all_preds, all_gts) * 100
        avg_ms = (total_time / max(total_samples, 1)) * 1000

        results = {
            "model": model_name,
            "num_samples": total_samples,
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
            "accuracy": round(acc, 4),
            "inference_ms": round(avg_ms, 2),
            "predictions": [
                {
                    "pred": p, "gt": g,
                    "cer": round(cer_score(p, g) * 100, 4),
                    "wer": round(wer_score(p, g) * 100, 4),
                }
                for p, g in zip(all_preds, all_gts)
            ],
        }

        self._save_results(results, model_name)
        logger.info(
            f"[{model_name}] CER={avg_cer:.2f}% WER={avg_wer:.2f}% "
            f"Acc={acc:.2f}% Inf={avg_ms:.1f}ms/sample"
        )
        return results

    def _seq2seq_infer(self, images: torch.Tensor) -> list[str]:
        """Greedy decoding for seq2seq model."""
        generated = self.model.generate(images, bos_id=1, eos_id=2)
        results = []
        for token_ids in generated:
            chars = [self.vocab[2].get(i, "") for i in token_ids
                     if self.vocab[2].get(i, "") not in ("<BLANK>", "<PAD>", "<EOS>", "<BOS>", "")]
            results.append("".join(chars))
        return results

    def _save_results(self, results: dict, model_name: str) -> None:
        """Save JSON and CSV result files."""
        out_dir = self.results_dir / "evaluations"
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON — full results
        json_path = out_dir / f"{model_name}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # CSV — per-sample predictions
        csv_path = out_dir / f"{model_name}_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pred", "gt", "cer", "wer"])
            writer.writeheader()
            writer.writerows(results["predictions"])

        logger.info(f"Results saved: {json_path}, {csv_path}")
