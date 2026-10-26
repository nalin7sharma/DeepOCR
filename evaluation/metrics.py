"""
OCR Evaluation Metrics for DeepOCR.

Implements:
  - CER  (Character Error Rate)  = edit_distance(pred_chars, gt_chars) / len(gt_chars)
  - WER  (Word Error Rate)       = edit_distance(pred_words, gt_words)  / len(gt_words)
  - Recognition Accuracy         = fraction of samples with exact string match
  - Inference Time               = measured externally and passed in

Edit distance uses Wagner-Fischer dynamic programming (O(n*m) time).
"""
from __future__ import annotations


def _edit_distance(a: list, b: list) -> int:
    """Wagner-Fischer edit distance (insertions, deletions, substitutions).

    Args:
        a: Reference sequence.
        b: Hypothesis sequence.

    Returns:
        Minimum edit distance (int).
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def cer_score(pred: str, gt: str) -> float:
    """Compute Character Error Rate.

    CER = edit_distance(pred_chars, gt_chars) / max(len(gt_chars), 1)

    Args:
        pred: Predicted text string.
        gt:   Ground-truth text string.

    Returns:
        CER as float in [0, inf).  Perfect match -> 0.0.
    """
    if not gt:
        return 0.0 if not pred else 1.0
    dist = _edit_distance(list(pred), list(gt))
    return dist / len(gt)


def wer_score(pred: str, gt: str) -> float:
    """Compute Word Error Rate.

    WER = edit_distance(pred_words, gt_words) / max(len(gt_words), 1)

    Args:
        pred: Predicted text string.
        gt:   Ground-truth text string.

    Returns:
        WER as float in [0, inf).
    """
    pred_words = pred.split()
    gt_words = gt.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    dist = _edit_distance(pred_words, gt_words)
    return dist / len(gt_words)


def recognition_accuracy(preds: list[str], gts: list[str]) -> float:
    """Compute exact-match recognition accuracy.

    Args:
        preds: List of predicted strings.
        gts:   List of ground-truth strings (same length).

    Returns:
        Fraction of samples where pred == gt (case-sensitive), in [0, 1].
    """
    if not gts:
        return 0.0
    matches = sum(p.strip() == g.strip() for p, g in zip(preds, gts))
    return matches / len(gts)


def batch_cer(preds: list[str], gts: list[str]) -> float:
    """Average CER over a list of (prediction, ground-truth) pairs."""
    if not gts:
        return 0.0
    return sum(cer_score(p, g) for p, g in zip(preds, gts)) / len(gts)


def batch_wer(preds: list[str], gts: list[str]) -> float:
    """Average WER over a list of (prediction, ground-truth) pairs."""
    if not gts:
        return 0.0
    return sum(wer_score(p, g) for p, g in zip(preds, gts)) / len(gts)
