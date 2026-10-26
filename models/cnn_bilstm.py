"""
Baseline OCR Model: CNN Feature Extractor + Bidirectional LSTM + CTC Loss.

Architecture:
  Input: [B, 1, H, W]  (grayscale, H=64 by default)
  CNN backbone: 4 conv blocks -> feature map [B, C, H//16, W//4]
  Reshape to sequence: [W//4, B, C * H//16]
  BiLSTM: 2 layers -> [W//4, B, 2*lstm_hidden]
  Linear projection: -> [W//4, B, num_classes]
  CTC decode at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool (optional)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        pool: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        if self.pool:
            x = self.pool(x)
        return x


class CNNBiLSTM(nn.Module):
    """CNN + Bidirectional LSTM OCR model trained with CTC loss.

    Args:
        num_classes: Size of the output vocabulary (including CTC blank at idx 0).
        cnn_channels: List of output channels for each conv block.
        lstm_hidden: Hidden size of each LSTM direction.
        lstm_layers: Number of stacked LSTM layers.
        lstm_dropout: Dropout between LSTM layers.
        image_height: Expected input image height (must match dataset setting).
    """

    def __init__(
        self,
        num_classes: int,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        image_height: int = 64,
    ) -> None:
        super().__init__()
        channels = cnn_channels or [64, 128, 256, 256]

        # ── CNN Feature Extractor ──────────────────────────────────────────
        # Each MaxPool(2,2) halves H and W.
        # 4 pools -> H/16, but we pool H only twice (H/4) and W twice (W/4)
        # to keep a reasonable sequence length.
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            cnn_layers.append(ConvBlock(in_ch, out_ch, kernel=3))
            if i < 2:
                # First two blocks: pool both H and W
                cnn_layers.append(nn.MaxPool2d((2, 2)))
            else:
                # Later blocks: pool only height to preserve sequence width
                cnn_layers.append(nn.MaxPool2d((2, 1)))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute the flattened feature size per time-step.
        # After 2x (2,2) pools: H/4.  After 2x (2,1) pools: H/4/2/2 = H/16
        # For image_height=64 -> feature_h = 64/4 = 16 (first 2 pools)
        #   then (2,1) pools halve H again -> 4
        # Measured with a dummy pass so any channel list / input height works.
        with torch.no_grad():
            self._feature_h = self.cnn(torch.zeros(1, 1, image_height, 64)).shape[2]
        self.image_height = image_height
        self._feature_ch = channels[-1]
        rnn_input_size = self._feature_h * self._feature_ch

        # ── Bidirectional LSTM ─────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,  # (T, B, C) convention for CTC
        )

        # ── Output projection ─────────────────────────────────────────────
        self.dropout = nn.Dropout(lstm_dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 1, H, W].

        Returns:
            Log-softmax logits [T, B, num_classes] where T = W_reduced.
        """
        # CNN features: [B, C, h, w]
        features = self.cnn(x)

        B, C, h, w = features.shape
        # Merge channel and height dims -> [B, C*h, w]
        features = features.view(B, C * h, w)
        # Permute to [w, B, C*h] = [T, B, features] for RNN
        features = features.permute(2, 0, 1)

        # RNN: [T, B, 2*hidden]
        rnn_out, _ = self.rnn(features)

        # Linear: [T, B, num_classes]
        logits = self.fc(self.dropout(rnn_out))
        return F.log_softmax(logits, dim=2)


def build_cnn_bilstm(
    num_classes: int,
    config: dict | None = None,
    image_height: int = 64,
) -> CNNBiLSTM:
    """Convenience factory that builds CNNBiLSTM from a config dict.

    Args:
        num_classes: Vocabulary size including CTC blank.
        config: Dict matching keys in configs/model.yaml under `cnn_bilstm`.

    Returns:
        Initialised model.
    """
    cfg = config or {}
    return CNNBiLSTM(
        num_classes=num_classes,
        cnn_channels=cfg.get("cnn_channels", [64, 128, 256, 256]),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        lstm_dropout=cfg.get("lstm_dropout", 0.3),
        image_height=cfg.get("image_height", image_height),
    )
