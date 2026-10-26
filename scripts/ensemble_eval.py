#!/usr/bin/env python
"""
Evaluate multi-model word voting (ROVER) on the held-out test predictions.

Each line is read by all five recognisers; their words are aligned and voted
on (see postprocessing/ensemble.py). A model's vote = its weight x its
confidence on that line, so a word several models agree on, confidently,
wins over a single model's guess.

Reads the per-line predictions written by scripts/evaluate.py (the same 600
IAM and 600 synthetic lines for every model), so no model is re-run. Model
weights are chosen by grid search with 2-fold cross-validation: weights tuned on
one half of the lines are scored only on the other half, so the reported
ensemble numbers are never measured on the lines used to tune them. Final
production weights are then fitted on all lines and stored for the web engine.

Usage:
    python scripts/ensemble_eval.py               # objective: overall CER (handwriting-weighted)
    python scripts/ensemble_eval.py --balanced    # objective: equal relative gain on both text types
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.line_image import normalize_transcription
from evaluation.metrics import cer_score, wer_score
from postprocessing.ensemble import Hypothesis, rover

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"]
SETS = ["iam", "synthetic"]
GRID = [0, 1, 2, 3]   # weights for the four other models; TrOCR fine-tuned (strongest) is fixed at PIVOT
PIVOT = 3
BALANCED = "--balanced" in sys.argv


def load() -> list[dict]:
    rows: list[dict] = []
    for set_name in SETS:
        per_model = {}
        for model in MODELS:
            path = ROOT / "results" / "evaluation" / "predictions" / f"{model}_{set_name}.csv"
            with open(path, encoding="utf-8") as handle:
                per_model[model] = {r["image_path"]: r for r in csv.DictReader(handle)}
        for image_path, base in per_model[MODELS[0]].items():
            rows.append({
                "set": set_name,
                "truth": normalize_transcription(base["ground_truth"]),
                "preds": {m: normalize_transcription(per_model[m][image_path]["prediction"]) for m in MODELS},
                "conf": {m: float(per_model[m][image_path]["confidence"] or 0) for m in MODELS},
            })
    return rows


def score(texts: list[str], rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {
        "cer": round(100 * sum(cer_score(t, r["truth"]) for t, r in zip(texts, rows)) / n, 3),
        "wer": round(100 * sum(wer_score(t, r["truth"]) for t, r in zip(texts, rows)) / n, 3),
        "accuracy": round(100 * sum(t == r["truth"] for t, r in zip(texts, rows)) / n, 3),
        "samples": len(rows),
    }


def ensemble(rows: list[dict], weights: dict[str, float]) -> list[str]:
    """Each model votes with weight x its confidence on that line."""
    return [rover([Hypothesis(r["preds"][m], m, b * r["conf"][m]) for m, b in weights.items() if b > 0])[0]
            for r in rows]


def objective(rows: list[dict], weights: dict[str, float], baseline: dict[str, float]) -> float:
    if not BALANCED:
        return score(ensemble(rows, weights), rows)["cer"]
    total = 0.0
    for name in SETS:
        part = [r for r in rows if r["set"] == name]
        total += score(ensemble(part, weights), part)["cer"] / baseline[name]
    return total


def tune(rows: list[dict]) -> dict[str, float]:
    baseline = {}
    for name in SETS:
        part = [r for r in rows if r["set"] == name]
        baseline[name] = max(1e-6, score([r["preds"]["trocr_ft"] for r in part], part)["cer"])
    best, best_value = None, float("inf")
    for combo in itertools.product(GRID, repeat=len(MODELS) - 1):
        weights = {"trocr_ft": PIVOT, **dict(zip(MODELS[1:], combo))}
        value = objective(rows, weights, baseline)
        if value < best_value:
            best, best_value = weights, value
    return best


def main() -> None:
    rows = load()
    folds = [rows[0::2], rows[1::2]]
    held_out: dict[int, str] = {}
    fold_weights = []
    for k in (0, 1):
        weights = tune(folds[k])
        fold_weights.append(weights)
        for offset, text in enumerate(ensemble(folds[1 - k], weights)):
            held_out[2 * offset + (1 - k)] = text
        print(f"fold {k}: tuned weights {weights}", flush=True)
    cv_texts = [held_out[i] for i in range(len(rows))]
    final_weights = tune(rows)
    print(f"final weights (all lines): {final_weights}")

    report = {
        "method": "ROVER word voting, vote = model weight x line confidence",
        "objective": "balanced relative CER" if BALANCED else "overall CER",
        "lines": len(rows),
        "fold_weights": fold_weights,
        "weights": final_weights,
        "sets": {},
    }
    for set_name in SETS:
        idx = [i for i, r in enumerate(rows) if r["set"] == set_name]
        subset = [rows[i] for i in idx]
        entry = {m: score([r["preds"][m] for r in subset], subset) for m in MODELS}
        entry["ensemble_cv"] = score([cv_texts[i] for i in idx], subset)
        entry["oracle_best_line"] = score(
            [min(r["preds"].values(), key=lambda t: cer_score(t, r["truth"])) for r in subset], subset)
        report["sets"][set_name] = entry
        print(f"\n{set_name}:")
        for name, value in entry.items():
            print(f"  {name:17s} CER {value['cer']:6.2f}%  WER {value['wer']:6.2f}%  Acc {value['accuracy']:6.2f}%")

    out = ROOT / "results" / "evaluation" / "ensemble.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Merge the cross-validated ensemble scores into the shared results summary.
    from scripts.evaluate import write_report_tables

    summary_path = ROOT / "results" / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    speed = sum(summary["models"][m][s]["ms_per_line"] for m in MODELS for s in SETS) / len(SETS)
    summary["models"]["ensemble"] = {
        "_info": {"label": "Consensus vote (5 models)", "architecture": "ROVER word voting over all engines",
                  "type": "ensemble of our models and the pretrained ones", "weights": final_weights},
        **{s: {**report["sets"][s]["ensemble_cv"], "ms_per_line": round(speed, 1)} for s in SETS},
    }
    notes = summary.setdefault("notes", [])
    note = ("The consensus vote is scored with 2-fold cross-validation: weights tuned on one half of "
            "the test lines are measured on the other half.")
    if note not in notes:
        notes.append(note)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report_tables(summary, ROOT / "results")
    print(f"\nSaved {out} and updated {summary_path}")


if __name__ == "__main__":
    main()
