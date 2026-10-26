"""
Transformer-Based OCR Model for DeepOCR.

Two operation modes:
  1. Pretrained (default): Wraps microsoft/trocr-base-handwritten from HuggingFace.
     - Requires: pip install transformers
     - The model is encoder-decoder (ViT encoder + GPT-2-like decoder).
     - Supports fine-tuning on custom data.

  2. Custom (use_pretrained=False): A lightweight Vision Transformer encoder
     paired with a Transformer decoder, trained from scratch.
     - Suitable for experimentation without HuggingFace dependency.

Usage:
    model = build_transformer_ocr(num_classes, config, processor)
    logits = model(pixel_values, decoder_input_ids)   # training
    texts  = model.generate(pixel_values, processor)  # inference
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Lightweight Custom ViT + Transformer Decoder (no HuggingFace required)
# ===========================================================================

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, image_height: int, image_width: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_h = image_height // patch_size
        self.n_patches_w = image_width // patch_size
        self.num_patches = self.n_patches_h * self.n_patches_w
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] -> [B, num_patches, embed_dim]."""
        x = self.proj(x)          # [B, embed_dim, H/p, W/p]
        x = x.flatten(2)          # [B, embed_dim, num_patches]
        return x.transpose(1, 2)  # [B, num_patches, embed_dim]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, embed_dim: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CustomViTOCR(nn.Module):
    """Lightweight Vision Transformer OCR (custom, trained from scratch).

    Args:
        num_classes: Output vocabulary size.
        image_height, image_width: Input dimensions.
        patch_size: Patch size for ViT.
        embed_dim: Embedding dimension.
        num_heads: Attention heads.
        encoder_layers: Number of ViT encoder layers.
        decoder_layers: Number of Transformer decoder layers.
        dropout: Dropout rate.
        max_target_length: Maximum sequence length for decoder.
    """

    def __init__(
        self,
        num_classes: int,
        image_height: int = 64,
        image_width: int = 512,
        patch_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        max_target_length: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_target_length = max_target_length

        # Patch embedding + CLS token + positional encoding
        self.patch_embed = PatchEmbedding(image_height, image_width, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_pos_enc = nn.Embedding(num_patches + 1, embed_dim)

        # ViT Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 4, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # Token embedding + positional encoding for decoder
        self.token_embed = nn.Embedding(num_classes, embed_dim)
        self.dec_pos_enc = PositionalEncoding(embed_dim, max_len=max_target_length + 1, dropout=dropout)

        # Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 4, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image patches.

        Args:
            pixel_values: [B, 1, H, W]

        Returns:
            Encoded memory [B, num_patches+1, embed_dim]
        """
        B = pixel_values.size(0)
        patches = self.patch_embed(pixel_values)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
        x = torch.cat([cls, patches], dim=1)       # [B, N+1, D]
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.enc_pos_enc(pos)
        return self.encoder(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive decode step.

        Args:
            tgt_ids: [B, T] token indices.
            memory: Encoder output [B, N, D].
            tgt_mask: Causal attention mask [T, T].

        Returns:
            Logits [B, T, num_classes].
        """
        tgt = self.dec_pos_enc(self.token_embed(tgt_ids))  # [B, T, D]
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_proj(out)

    def forward(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass (teacher forcing).

        Args:
            pixel_values: [B, 1, H, W]
            decoder_input_ids: [B, T] target token ids (shifted right).

        Returns:
            Logits [B, T, num_classes].
        """
        T = decoder_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=pixel_values.device)
        memory = self.encode(pixel_values)
        return self.decode(decoder_input_ids, memory, tgt_mask=tgt_mask)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> list[list[int]]:
        """Greedy autoregressive generation.

        Args:
            pixel_values: [B, 1, H, W]
            bos_id: Begin-of-sequence token index.
            eos_id: End-of-sequence token index.

        Returns:
            List of generated token id sequences (one per batch item).
        """
        B = pixel_values.size(0)
        memory = self.encode(pixel_values)
        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
        results = [[] for _ in range(B)]
        finished = [False] * B

        for _ in range(self.max_target_length):
            T = cur_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=pixel_values.device)
            logits = self.decode(cur_ids, memory, tgt_mask=tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)  # [B]

            for b in range(B):
                if not finished[b]:
                    tok = next_token[b].item()
                    if tok == eos_id:
                        finished[b] = True
                    else:
                        results[b].append(tok)

            if all(finished):
                break
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)

        return results


# ===========================================================================
# TrOCR Wrapper (HuggingFace pretrained)
# ===========================================================================

class TrOCRWrapper(nn.Module):
    """Wrapper around HuggingFace TrOCR (microsoft/trocr-base-handwritten).

    This class wraps the pretrained TrOCR model for fine-tuning or
    zero-shot inference.  Requires:
        pip install transformers

    If the transformers library is not installed, a clear error is raised.

    Args:
        model_name: HuggingFace model identifier.
        num_classes: Not used directly (TrOCR uses its own vocab).
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten") -> None:
        super().__init__()
        try:
            from transformers import VisionEncoderDecoderModel, TrOCRProcessor
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self._processor_name = model_name
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers is required for TrOCR. "
                "Install it with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TrOCR model '{model_name}'. "
                f"Check your internet connection or model name. Error: {e}"
            ) from e

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        """Training forward pass."""
        return self.model(pixel_values=pixel_values, labels=labels)

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, processor=None) -> list[str]:
        """Generate text predictions.

        Args:
            pixel_values: [B, 3, H, W] RGB images (TrOCR expects RGB).
            processor: TrOCRProcessor for decoding token ids to text.

        Returns:
            List of predicted strings.
        """
        generated = self.model.generate(pixel_values)
        if processor is not None:
            return processor.batch_decode(generated, skip_special_tokens=True)
        return [str(g.tolist()) for g in generated]


# ===========================================================================
# TransformerOCR: unified interface
# ===========================================================================

class TransformerOCR(nn.Module):
    """Unified Transformer OCR interface.

    Selects between the custom ViT+decoder model and the TrOCR wrapper
    depending on the config.

    Args:
        num_classes: Vocabulary size.
        config: Dict with transformer_ocr settings from model.yaml.
    """

    def __init__(self, num_classes: int, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.use_pretrained = cfg.get("use_pretrained", True)

        if self.use_pretrained:
            model_name = cfg.get("encoder", "microsoft/trocr-base-handwritten")
            self.model = TrOCRWrapper(model_name)
            self.is_trocr = True
        else:
            vit_cfg = cfg.get("custom_vit", {})
            self.model = CustomViTOCR(
                num_classes=num_classes,
                patch_size=vit_cfg.get("patch_size", 16),
                embed_dim=vit_cfg.get("embed_dim", 512),
                num_heads=vit_cfg.get("num_heads", 8),
                encoder_layers=vit_cfg.get("encoder_layers", 6),
                decoder_layers=vit_cfg.get("decoder_layers", 6),
                dropout=vit_cfg.get("dropout", 0.1),
                max_target_length=cfg.get("max_target_length", 128),
            )
            self.is_trocr = False
        self.num_classes = num_classes

    def forward(self, pixel_values, decoder_input_ids=None, labels=None):
        if self.is_trocr:
            return self.model(pixel_values, labels=labels)
        return self.model(pixel_values, decoder_input_ids)

    def generate(self, pixel_values, processor=None, bos_id=1, eos_id=2):
        if self.is_trocr:
            return self.model.generate(pixel_values, processor=processor)
        return self.model.generate(pixel_values, bos_id=bos_id, eos_id=eos_id)


def build_transformer_ocr(
    num_classes: int,
    config: dict | None = None,
) -> TransformerOCR:
    """Build TransformerOCR from config dict."""
    return TransformerOCR(num_classes=num_classes, config=config)
