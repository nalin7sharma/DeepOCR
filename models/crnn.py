"""
CRNN: Convolutional Recurrent Neural Network for Sequence Recognition.
Based on Shi et al. (2016) "An End-to-End Trainable Neural Network for
Image-based Sequence Recognition and Its Application to Scene Text Recognition".

Architecture:
  VGG-inspired deep CNN (7 conv layers) -> map-to-sequence -> 2-layer BiLSTM -> CTC

The paper uses channels [64, 128, 256, 256, 512, 512, 512] on 32 px tall
inputs.  Channel widths and input height are configurable so the same
topology can be trained on a CPU with narrower layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_CHANNELS = [64, 128, 256, 256, 512, 512, 512]


class CRNN(nn.Module):
    """CRNN model for OCR (Shi et al. 2016 topology).

    Args:
        num_classes: Output vocabulary size (includes CTC blank at 0).
        lstm_hidden: LSTM hidden units per direction.
        lstm_layers: Kept for config compatibility (the paper uses 2 BiLSTMs).
        leaky_relu: Use LeakyReLU instead of ReLU in CNN.
        cnn_channels: Seven channel widths for the seven conv layers.
        image_height: Input height in pixels.
        dropout: Dropout applied between the two BiLSTM layers.
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        leaky_relu: bool = False,
        cnn_channels: list[int] | None = None,
        image_height: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_height = image_height
        c = list(cnn_channels or PAPER_CHANNELS)
        if len(c) != 7:
            raise ValueError("CRNN expects exactly seven conv channel widths.")

        def act():
            return nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)

        def conv_bn(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(in_c, out_c, k, stride=s, padding=p), nn.BatchNorm2d(out_c), act())

        # ── CNN backbone (paper configuration) ────────────────────────────
        self.cnn = nn.Sequential(
            conv_bn(1, c[0]),
            nn.MaxPool2d(2, 2),               # H/2, W/2
            conv_bn(c[0], c[1]),
            nn.MaxPool2d(2, 2),               # H/4, W/4
            conv_bn(c[1], c[2]),
            conv_bn(c[2], c[3]),
            nn.MaxPool2d((2, 1)),             # H/8, W/4
            conv_bn(c[3], c[4]),
            conv_bn(c[4], c[5]),
            nn.MaxPool2d((2, 1)),             # H/16, W/4
            conv_bn(c[5], c[6], k=2, s=1, p=0),  # H/16-1, W/4-1
        )

        # Feature height depends on the input height; measure it once.
        with torch.no_grad():
            feature_h = self.cnn(torch.zeros(1, 1, image_height, 64)).shape[2]
        self._rnn_input = c[6] * feature_h

        self.rnn = nn.Sequential(
            BidirectionalLSTM(self._rnn_input, lstm_hidden, lstm_hidden),
            nn.Dropout(dropout),
            BidirectionalLSTM(lstm_hidden, lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 1, H, W] grayscale images.

        Returns:
            [T, B, num_classes] log-softmax probabilities.
        """
        conv = self.cnn(x)               # [B, C, h, w]
        B, C, h, w = conv.shape
        conv = conv.view(B, C * h, w)    # [B, C*h, w]
        conv = conv.permute(2, 0, 1)     # [w, B, C*h] = [T, B, features]
        output = self.rnn(conv)          # [T, B, num_classes]
        return F.log_softmax(output, dim=2)


class BidirectionalLSTM(nn.Module):
    """A single bidirectional LSTM layer with linear projection."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        T, B, _ = out.shape
        out = out.view(T * B, -1)
        out = self.fc(out)
        return out.view(T, B, -1)


def build_crnn(num_classes: int, config: dict | None = None, image_height: int = 64) -> CRNN:
    """Build CRNN from config dict."""
    cfg = config or {}
    return CRNN(
        num_classes=num_classes,
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        cnn_channels=cfg.get("cnn_channels", PAPER_CHANNELS),
        image_height=cfg.get("image_height", image_height),
        dropout=cfg.get("dropout", 0.2),
    )
