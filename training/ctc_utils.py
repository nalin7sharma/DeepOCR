"""CTC decoding utilities for DeepOCR.

Provides:
  - ctc_greedy_decode: Fast greedy CTC decoding.
  - ctc_beam_decode:   Beam-search CTC decoding (pure Python fallback;
                       use ctcdecode library for production speed).
"""
from __future__ import annotations

import torch
import numpy as np


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    idx2char: dict[int, str],
    blank_idx: int = 0,
) -> list[str]:
    """Greedy CTC decoding: argmax at each timestep then collapse.

    Args:
        log_probs: Log-softmax output [T, B, C] from CTC model.
        idx2char:  Index-to-character mapping.
        blank_idx: Index of the CTC blank token.

    Returns:
        List of decoded strings, one per batch item.
    """
    # Take argmax over class dimension -> [T, B]
    preds = log_probs.argmax(dim=2)  # [T, B]
    preds = preds.permute(1, 0)      # [B, T]

    results: list[str] = []
    for pred_seq in preds:
        chars = []
        prev = blank_idx
        for idx in pred_seq.tolist():
            if idx != blank_idx and idx != prev:
                char = idx2char.get(idx, "")
                if char and char != "<BLANK>":
                    chars.append(char)
            prev = idx
        results.append("".join(chars))
    return results


def ctc_beam_decode(
    log_probs: torch.Tensor,
    idx2char: dict[int, str],
    blank_idx: int = 0,
    beam_width: int = 5,
) -> list[str]:
    """Pure-Python beam-search CTC decoding.

    For production use, replace with the ctcdecode library for speed.

    Args:
        log_probs:  Log-softmax output [T, B, C].
        idx2char:   Index to character mapping.
        blank_idx:  CTC blank token index.
        beam_width: Number of beams to maintain.

    Returns:
        List of best decoded strings per batch item.
    """
    probs = torch.exp(log_probs).cpu().numpy()  # [T, B, C]
    T, B, C = probs.shape
    results = []

    for b in range(B):
        prob_seq = probs[:, b, :]  # [T, C]

        # Each beam: (prefix_string, last_char_idx, prob)
        beams: list[tuple[str, int, float]] = [("", blank_idx, 1.0)]

        for t in range(T):
            new_beams: dict[str, tuple[int, float]] = {}

            for prefix, last_c, beam_prob in beams:
                for c in range(C):
                    p = float(prob_seq[t, c])
                    if p < 1e-10:
                        continue

                    if c == blank_idx:
                        # Blank: keep prefix, reset last char
                        key = prefix
                        new_last = blank_idx
                    elif c == last_c:
                        # Same as last: only extend if last was blank
                        key = prefix
                        new_last = c
                    else:
                        char = idx2char.get(c, "")
                        if not char or char == "<BLANK>":
                            continue
                        key = prefix + char
                        new_last = c

                    new_prob = beam_prob * p
                    if key in new_beams:
                        old_last, old_prob = new_beams[key]
                        if new_prob > old_prob:
                            new_beams[key] = (new_last, new_prob)
                    else:
                        new_beams[key] = (new_last, new_prob)

            # Keep top beam_width beams
            sorted_beams = sorted(new_beams.items(), key=lambda x: -x[1][1])
            beams = [(k, v[0], v[1]) for k, v in sorted_beams[:beam_width]]

        best = beams[0][0] if beams else ""
        results.append(best)

    return results


def compute_ctc_input_lengths(
    images: torch.Tensor,
    time_reduction_factor: int = 4,
) -> torch.Tensor:
    """Compute CTC input lengths (time steps) from image widths.

    The CNN reduces the spatial width by a fixed factor.  This function
    calculates the actual sequence length for each image in the batch,
    accounting for padding.

    Args:
        images:  [B, C, H, W] image batch.
        time_reduction_factor: How much the CNN reduces the width dimension.

    Returns:
        LongTensor of shape [B] with sequence lengths.
    """
    widths = images.size(3)  # all same width due to collate padding
    lengths = torch.full((images.size(0),), widths // time_reduction_factor, dtype=torch.long)
    return lengths
