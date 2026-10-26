"""
Generic Trainer for DeepOCR models.

Supports:
  - CTC-based models (CNN+BiLSTM, CRNN)
  - Sequence-to-sequence models (Transformer OCR) — cross-entropy loss
  - LR schedulers: cosine, step, plateau
  - Gradient clipping
  - Early stopping
  - Per-epoch checkpoint saving + best-model selection
  - History tracking for training curves
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluation.metrics import cer_score, wer_score
from training.ctc_utils import ctc_greedy_decode, compute_ctc_input_lengths
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.visualization import plot_training_curves


logger = get_logger(__name__)


class Trainer:
    """Manages the train/validate loop for any DeepOCR model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Training config dict (from configs/training.yaml).
        checkpoint_dir: Directory to save checkpoints.
        results_dir: Directory to save logs and plots.
        vocab: (chars, char2idx, idx2char) tuple — used for CTC decoding.
        model_type: "ctc" | "seq2seq" — determines loss function.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: str | Path = "checkpoints",
        results_dir: str | Path = "results",
        vocab: tuple | None = None,
        model_type: str = "ctc",
        device: torch.device | str = "auto",
        meta: dict | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config.get("training", config)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.model_type = model_type
        self.vocab = vocab
        # Architecture/charset/input-size metadata stored in every checkpoint so
        # inference can rebuild the exact model (see models/recognizer.py).
        self.meta = meta or {}
        self.epoch_seconds: list[float] = []
        self.started_at = time.time()

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Optimiser
        lr = self.cfg.get("learning_rate", 0.001)
        wd = self.cfg.get("weight_decay", 0.0001)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # LR Scheduler
        sched_name = self.cfg.get("lr_scheduler", "cosine")
        n_epochs = self.cfg.get("epochs", 50)
        if sched_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        elif sched_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.get("lr_step_size", 10),
                gamma=self.cfg.get("lr_gamma", 0.5),
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        self.plateau_scheduler = sched_name == "plateau"

        # Loss
        if model_type == "ctc":
            self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_cer": [], "val_cer": [], "val_wer": []
        }
        self.best_metric = float("inf")
        self.early_stop_counter = 0

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Training history dict.
        """
        n_epochs = self.cfg.get("epochs", 50)
        patience = self.cfg.get("early_stopping_patience", 10)
        save_every = self.cfg.get("save_every_n_epochs", 5)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss, train_cer = self._train_epoch(epoch)
            val_loss, val_cer, val_wer = self._validate_epoch(epoch)
            elapsed = time.time() - t0

            # Step scheduler
            if self.plateau_scheduler:
                self.scheduler.step(val_cer)
            else:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_cer"].append(train_cer)
            self.history["val_cer"].append(val_cer)
            self.history["val_wer"].append(val_wer)

            # Calculate ETA
            remaining_epochs = n_epochs - epoch
            eta_seconds = elapsed * remaining_epochs
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            lr = self.optimizer.param_groups[0]["lr"]
            self.history.setdefault("lr", []).append(lr)
            self.epoch_seconds.append(elapsed)
            logger.info(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"TrainLoss={train_loss:.4f} TrainCER={train_cer:.4f} | "
                f"ValLoss={val_loss:.4f} ValCER={val_cer:.4f} ValWER={val_wer:.4f} | "
                f"LR={lr:.6f} | {elapsed:.1f}s | ETA: {eta_str}"
            )

            # Save periodic checkpoint
            if epoch % save_every == 0:
                self._save(epoch, val_cer, is_best=False)

            # Check best model + early stopping
            is_best = val_cer < self.best_metric
            if is_best:
                self.best_metric = val_cer
                self._save(epoch, val_cer, is_best=True)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            self._write_history(epoch, n_epochs, finished=False)
            if self.early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

            # Optional wall-clock budget (useful for CPU-only training runs)
            budget_minutes = self.cfg.get("max_minutes")
            if budget_minutes:
                spent = time.time() - self.started_at
                mean_epoch = sum(self.epoch_seconds) / len(self.epoch_seconds)
                if spent + mean_epoch > budget_minutes * 60:
                    logger.info(f"Stopping after epoch {epoch}: {budget_minutes}-minute budget reached.")
                    break

        self._write_history(len(self.history["val_cer"]), n_epochs, finished=True)
        model_name = self.cfg.get("model", "model")
        plot_path = self.results_dir / "plots" / f"{model_name}_training_curves.png"
        plot_training_curves(self.history, plot_path, title=f"{model_name} training curves")
        logger.info(f"Training curves saved to {plot_path}")
        return self.history

    def _write_history(self, epoch: int, n_epochs: int, finished: bool) -> None:
        """Persist the live training history (read by the web dashboard)."""
        model_name = self.cfg.get("model", "model")
        out = self.results_dir / "training" / f"{model_name}_history.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_name,
            "epoch": epoch,
            "epochs_planned": n_epochs,
            "finished": finished,
            "best_val_cer": None if self.best_metric == float("inf") else self.best_metric,
            "epoch_seconds": self.epoch_seconds,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            "meta": self.meta,
            "history": self.history,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            (avg_loss, avg_cer) for training set.
        """
        self.model.train()
        total_loss = 0.0
        all_pred: list[str] = []
        all_gt: list[str] = []
        clip = self.cfg.get("gradient_clip", 5.0)

        for batch in self.train_loader:
            images = batch["image"].to(self.device)           # [B, 1, H, W]
            labels = batch["label"].to(self.device)           # [B, T_label]
            label_lengths = batch["label_length"].to(self.device)  # [B]
            label_texts = batch["label_text"]

            self.optimizer.zero_grad()

            if self.model_type == "ctc":
                log_probs = self.model(images)  # [T, B, C]
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
                # Concatenate unpadded targets for 1D CTCLoss target format
                targets_unpadded = torch.cat([labels[b, :label_lengths[b]] for b in range(B)])
                loss = self.criterion(
                    log_probs,
                    targets_unpadded,
                    input_lengths.cpu(),
                    label_lengths.cpu(),
                )
                # Decode for metric
                preds = ctc_greedy_decode(log_probs.detach().cpu(), self.vocab[2])
            else:
                # seq2seq: teacher forcing
                # decoder_input = BOS + label[:-1]; target = label
                bos = torch.zeros(images.size(0), 1, dtype=torch.long, device=self.device)
                dec_input = torch.cat([bos, labels[:, :-1]], dim=1)
                logits = self.model(images, dec_input)  # [B, T, C]
                B2, T2, C2 = logits.shape
                loss = self.criterion(logits.view(B2 * T2, C2), labels.view(-1))
                preds = [self._decode_seq(logits[b]) for b in range(B2)]

            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.item()
            all_pred.extend(preds)
            all_gt.extend(label_texts)

        avg_loss = total_loss / max(len(self.train_loader), 1)
        avg_cer = float(sum(cer_score(p, g) for p, g in zip(all_pred, all_gt)) / max(len(all_pred), 1))
        return avg_loss, avg_cer

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Run one validation epoch.

        Returns:
            (avg_loss, avg_cer, avg_wer).
        """
        self.model.eval()
        total_loss = 0.0
        all_pred: list[str] = []
        all_gt: list[str] = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            label_lengths = batch["label_length"].to(self.device)
            label_texts = batch["label_text"]

            if self.model_type == "ctc":
                log_probs = self.model(images)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
                targets_unpadded = torch.cat([labels[b, :label_lengths[b]] for b in range(B)])
                loss = self.criterion(
                    log_probs,
                    targets_unpadded,
                    input_lengths.cpu(),
                    label_lengths.cpu(),
                )
                preds = ctc_greedy_decode(log_probs.cpu(), self.vocab[2])
            else:
                bos = torch.zeros(images.size(0), 1, dtype=torch.long, device=self.device)
                dec_input = torch.cat([bos, labels[:, :-1]], dim=1)
                logits = self.model(images, dec_input)
                B2, T2, C2 = logits.shape
                loss = self.criterion(logits.view(B2 * T2, C2), labels.view(-1))
                preds = [self._decode_seq(logits[b]) for b in range(B2)]

            total_loss += loss.item()
            all_pred.extend(preds)
            all_gt.extend(label_texts)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        avg_cer = float(sum(cer_score(p, g) for p, g in zip(all_pred, all_gt)) / max(len(all_pred), 1))
        avg_wer = float(sum(wer_score(p, g) for p, g in zip(all_pred, all_gt)) / max(len(all_pred), 1))
        return avg_loss, avg_cer, avg_wer

    def _decode_seq(self, logits: torch.Tensor) -> str:
        """Greedy decode for seq2seq output [T, C]."""
        ids = logits.argmax(dim=-1).tolist()
        chars = []
        for i in ids:
            char = self.vocab[2].get(i, "")
            if char in ("<EOS>", "<PAD>", "<BLANK>"):
                break
            if char:
                chars.append(char)
        return "".join(chars)

    def _save(self, epoch: int, val_cer: float, is_best: bool) -> None:
        model_name = self.cfg.get("model", "model")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_cer": val_cer,
            "history": self.history,
            "meta": self.meta,
        }
        fname = f"{model_name}_epoch{epoch:03d}.pth"
        ckpt_dir = self.checkpoint_dir / model_name
        save_checkpoint(state, ckpt_dir, fname, is_best=is_best)
