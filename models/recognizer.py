"""
Line recognisers used by evaluation, the CLI, Streamlit and the web API.

``LineRecognizer`` rebuilds a trained CTC model (CNN-BiLSTM or CRNN) purely
from the metadata stored in its checkpoint (architecture, charset, input
size), so inference never depends on the YAML configs matching the run that
produced the weights.

``TrOCRRecognizer`` wraps Microsoft's pretrained Transformer OCR
(ViT/DeiT encoder + Transformer decoder) for transfer-learning comparison.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from datasets.line_image import fit_width, normalize_line, resize_to_height, to_gray, to_tensor
from datasets.ocr_dataset import build_vocabulary

CTC_ARCHITECTURES = {"cnn_bilstm", "crnn"}


def _builders():
    from models.cnn_bilstm import build_cnn_bilstm
    from models.crnn import build_crnn

    return {"cnn_bilstm": build_cnn_bilstm, "crnn": build_crnn}


class LineRecognizer:
    """Trained CTC text-line recogniser restored from a DeepOCR checkpoint."""

    def __init__(self, model: torch.nn.Module, meta: dict, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.meta = meta
        self.device = torch.device(device)
        self.model_name: str = meta["model_name"]
        self.image_height: int = int(meta["image_height"])
        self.image_width: int = int(meta["image_width"])
        self.chars, self.char2idx, self.idx2char = build_vocabulary(meta["charset"])
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "LineRecognizer":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("meta")
        if not meta:
            raise ValueError(
                f"{path} has no model metadata; retrain with the current scripts/train.py."
            )
        if meta["model_name"] not in CTC_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture in checkpoint: {meta['model_name']}")
        num_classes = len(build_vocabulary(meta["charset"])[0])
        model = _builders()[meta["model_name"]](
            num_classes, dict(meta.get("model_cfg") or {}), image_height=int(meta["image_height"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        recognizer = cls(model, meta, device)
        recognizer.checkpoint_path = str(path)
        recognizer.epoch = checkpoint.get("epoch")
        recognizer.val_cer = checkpoint.get("val_cer")
        return recognizer

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        normalized = resize_to_height(normalize_line(image), self.image_height)
        # Training squeezes rare over-long lines to the training width; at inference
        # we allow up to twice that before squeezing so wide document lines stay legible.
        return fit_width(normalized, min(max(normalized.shape[1], 32), self.image_width * 2))

    @torch.no_grad()
    def recognize(self, images: Sequence[np.ndarray], batch_size: int = 16) -> list[tuple[str, float]]:
        """Recognise a list of line crops. Returns ``(text, confidence)`` per crop."""
        results: list[tuple[str, float]] = []
        prepared = [self._prepare(image) for image in images]
        order = sorted(range(len(prepared)), key=lambda i: prepared[i].shape[1])
        by_index: dict[int, tuple[str, float]] = {}
        with self._lock:
            for start in range(0, len(order), batch_size):
                chunk = order[start:start + batch_size]
                width = max(prepared[i].shape[1] for i in chunk)
                batch = torch.stack([to_tensor(fit_width(prepared[i], width)) for i in chunk]).to(self.device)
                log_probs = self.model(batch).cpu()  # [T, B, C]
                for offset, index in enumerate(chunk):
                    by_index[index] = self._decode(log_probs[:, offset, :])
        for index in range(len(prepared)):
            results.append(by_index[index])
        return results

    def _decode(self, log_probs: torch.Tensor) -> tuple[str, float]:
        """Greedy CTC decode with a confidence = mean max-probability of emitted chars."""
        probs = log_probs.exp()
        best_prob, best_idx = probs.max(dim=1)
        chars: list[str] = []
        scores: list[float] = []
        previous = 0
        for index, prob in zip(best_idx.tolist(), best_prob.tolist()):
            if index != 0 and index != previous:
                char = self.idx2char.get(index, "")
                if char and char != "<BLANK>":
                    chars.append(char)
                    scores.append(prob)
            previous = index
        text = " ".join("".join(chars).split())
        confidence = float(np.mean(scores)) if scores else 0.0
        return text, confidence

    def describe(self) -> dict:
        return {
            "architecture": self.model_name,
            "parameters": self.parameter_count,
            "input": f"{self.image_height}x{self.image_width}",
            "epoch": getattr(self, "epoch", None),
            "val_cer": getattr(self, "val_cer", None),
            "datasets": self.meta.get("datasets"),
        }


class TrOCRRecognizer:
    """Pretrained Transformer OCR (Microsoft TrOCR) line recogniser."""

    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten", device: str = "cpu",
                 num_beams: int = 1, max_new_tokens: int = 64) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.model_name = model_name
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
        self.device = torch.device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        gray = to_gray(image)
        # TrOCR was trained on dark ink over light paper.
        if float(np.median(gray)) < 127:
            gray = 255 - gray
        return np.stack([gray] * 3, axis=-1)

    @torch.no_grad()
    def recognize(self, images: Sequence[np.ndarray], batch_size: int = 8) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        with self._lock:
            for start in range(0, len(images), batch_size):
                chunk = [self._to_rgb(image) for image in images[start:start + batch_size]]
                pixel_values = self.processor(images=chunk, return_tensors="pt").pixel_values.to(self.device)
                output = self.model.generate(
                    pixel_values,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
                # Sequence confidence = geometric mean of the chosen tokens' probabilities.
                pad_id = self.model.generation_config.pad_token_id
                steps = []
                for t, step_scores in enumerate(output.scores):
                    chosen = output.sequences[:, t + 1]
                    steps.append(torch.log_softmax(step_scores.float(), dim=-1).gather(1, chosen[:, None]).squeeze(1))
                if steps:
                    log_probs = torch.stack(steps, dim=1)
                    mask = output.sequences[:, 1:1 + log_probs.shape[1]] != (pad_id if pad_id is not None else -1)
                    log_probs = torch.where(mask & torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
                    means = log_probs.sum(1) / mask.sum(1).clamp(min=1)
                else:
                    means = torch.zeros(len(texts))
                for text, mean in zip(texts, means.tolist()):
                    results.append((" ".join(text.split()), float(math.exp(mean))))
        return results

    def describe(self) -> dict:
        return {
            "architecture": "trocr",
            "parameters": self.parameter_count,
            "input": "384x384 RGB",
            "pretrained": self.model_name,
        }


class PaddleLineRecognizer:
    """PaddleOCR's pretrained English recogniser applied to pre-cropped lines (reference)."""

    def __init__(self, language: str = "en") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=False, lang=language, use_gpu=False, show_log=False)
        self._lock = threading.Lock()

    def recognize(self, images: Sequence[np.ndarray], batch_size: int = 16) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        with self._lock:
            for image in images:
                bgr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
                output = self._ocr.ocr(bgr, det=False, cls=False)
                try:
                    text, score = output[0][0]
                except (IndexError, TypeError, ValueError):
                    text, score = "", 0.0
                results.append((" ".join(str(text).split()), float(score)))
        return results

    def describe(self) -> dict:
        return {"architecture": "PP-OCR (SVTR/CRNN)", "parameters": None, "input": "48x320 RGB"}


def time_recognizer(recognizer, images: Sequence[np.ndarray]) -> tuple[list[tuple[str, float]], float]:
    """Run a recogniser and return results plus mean milliseconds per line."""
    started = time.perf_counter()
    results = recognizer.recognize(images)
    elapsed = (time.perf_counter() - started) * 1000
    return results, elapsed / max(1, len(images))
