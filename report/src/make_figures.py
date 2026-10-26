"""Draw every figure used in the project report from the project's own result files.

Usage (from the project root):
    python report/src/make_figures.py

Writes PNG files to report/figures/. Diagrams are drawn with matplotlib so the
report needs no external drawing tools; charts read results/ directly, so they
always match the numbers in the evaluation files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EVAL = ROOT / "results" / "evaluation"
TRAIN = ROOT / "results" / "training"

INK = "#101828"
MUTED = "#667085"
LINE = "#d0d5dd"
ACCENT = "#4f46e5"
SOFT = "#eef2ff"
COLORS = {
    "ensemble": "#4f46e5", "trocr_ft": "#f79009", "trocr": "#16b364",
    "crnn": "#7a5af8", "cnn_bilstm": "#0ba5ec", "paddle": "#667085",
}
LABELS = {
    "ensemble": "Consensus vote", "trocr_ft": "TrOCR (fine-tuned)", "trocr": "TrOCR (pretrained)",
    "crnn": "CRNN", "cnn_bilstm": "CNN-BiLSTM", "paddle": "PaddleOCR",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#eaecf0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, title, sub="", fill="#ffffff", edge=LINE, title_color=INK, size=10, sub_size=8.2, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=1.2, edgecolor=edge, facecolor=fill))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=title_color)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sub_size, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=title_color)


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- diagrams
def system_architecture():
    fig, ax = canvas(7.2, 4.3)
    stages = [
        ("1. Input", "PDF / image upload\napi/documents.py"),
        ("2. Preprocessing", "Deskew, denoise, CLAHE\napi/preprocess.py"),
        ("3. Text detection", "DBNet line polygons\nPP-OCR detector"),
        ("4. Recognition", "CNN-BiLSTM · CRNN · TrOCR\nmodels/, api/engines.py"),
        ("5. Consensus vote", "ROVER word voting\npostprocessing/ensemble.py"),
        ("6. Layout analysis", "Order, headings, tables\napi/structure.py"),
        ("7. Error correction", "Confusions, spelling\ntext_corrector.py"),
        ("8. Review & export", "Web studio, JSON / MD\nweb/, api/main.py"),
    ]
    w, h = 0.205, 0.22
    xs = [0.03, 0.27, 0.51, 0.755]
    for i, (title, sub) in enumerate(stages):
        row = i // 4
        col = i % 4
        x = xs[col]
        y = 0.62 if row == 0 else 0.2
        special = i == 4
        box(ax, x, y, w, h, title, sub, fill=SOFT if special else "#ffffff",
            edge=ACCENT if special else LINE, title_color=ACCENT if special else INK, size=9.6, sub_size=7.6)
        if col < 3:
            arrow(ax, x + w + 0.004, y + h / 2, xs[col + 1] - 0.004, y + h / 2)
    # Row 1 end -> row 2 start: down, left along the gutter, down.
    gutter = 0.52
    ax.plot([xs[3] + w / 2, xs[3] + w / 2], [0.62 - 0.012, gutter], color=MUTED, lw=1.3)
    ax.plot([xs[3] + w / 2, xs[0] + w / 2], [gutter, gutter], color=MUTED, lw=1.3)
    arrow(ax, xs[0] + w / 2, gutter, xs[0] + w / 2, 0.2 + h + 0.014)
    ax.text(0.5, 0.95, "DeepOCR processing pipeline", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.5, 0.07, "Stage 5 runs only with the Consensus engine; every other engine goes from recognition straight to layout analysis.",
            ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "fig_architecture")


def deployment_architecture():
    fig, ax = canvas(7.2, 3.3)
    box(ax, 0.02, 0.36, 0.2, 0.34, "Web browser", "React + TypeScript\nStudio · Compare\nResearch · How it works", sub_size=7.6)
    box(ax, 0.32, 0.12, 0.32, 0.8, "", fill="#f9fafb")
    ax.text(0.48, 0.86, "FastAPI service (api/main.py)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 0.35, 0.58, 0.26, 0.2, "REST endpoints", "/api/v1/jobs · /ocr · /engines\n/metrics · /samples", size=9, sub_size=7.2)
    box(ax, 0.35, 0.33, 0.26, 0.2, "Job worker thread", "progress · cancel · 1 at a time", size=9, sub_size=7.2)
    box(ax, 0.35, 0.16, 0.26, 0.13, "Engine registry", size=9)
    box(ax, 0.74, 0.66, 0.24, 0.24, "Results", "results/evaluation\nresults/training", sub_size=7.4)
    box(ax, 0.74, 0.34, 0.24, 0.26, "Trained models", "checkpoints/\nCNN-BiLSTM · CRNN\nTrOCR fine-tuned", sub_size=7.4)
    box(ax, 0.74, 0.04, 0.24, 0.24, "Pretrained", "TrOCR-small (HF)\nPaddleOCR PP-OCR", sub_size=7.4)
    arrow(ax, 0.225, 0.62, 0.345, 0.68, style="<|-|>")
    ax.text(0.27, 0.73, "HTTP / JSON", fontsize=7.5, color=MUTED, ha="center")
    arrow(ax, 0.48, 0.58, 0.48, 0.53)
    arrow(ax, 0.48, 0.33, 0.48, 0.29)
    arrow(ax, 0.615, 0.69, 0.735, 0.78)
    ax.text(0.675, 0.77, "metrics", fontsize=7, color=MUTED, ha="center", rotation=30)
    arrow(ax, 0.615, 0.23, 0.735, 0.46)
    arrow(ax, 0.615, 0.2, 0.735, 0.16)
    save(fig, "fig_deployment")


def use_case():
    fig, ax = canvas(6.4, 5.6)
    # Actor.
    cx, cy = 0.12, 0.5
    ax.add_patch(plt.Circle((cx, cy + 0.13), 0.03, fill=False, color=INK, lw=1.4))
    ax.plot([cx, cx], [cy + 0.1, cy - 0.02], color=INK, lw=1.4)
    ax.plot([cx - 0.045, cx + 0.045], [cy + 0.06, cy + 0.06], color=INK, lw=1.4)
    ax.plot([cx, cx - 0.04], [cy - 0.02, cy - 0.1], color=INK, lw=1.4)
    ax.plot([cx, cx + 0.04], [cy - 0.02, cy - 0.1], color=INK, lw=1.4)
    ax.text(cx, cy - 0.16, "User\n(student, clerk,\nresearcher)", ha="center", va="top", fontsize=8.5, color=INK)
    ax.add_patch(FancyBboxPatch((0.38, 0.02), 0.58, 0.94, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.2, edgecolor=LINE, facecolor="#fcfcfd"))
    ax.text(0.67, 0.93, "DeepOCR system", ha="center", fontsize=10, fontweight="bold", color=INK)
    cases = [
        "Upload a document or pick a sample", "Choose engine, cleanup and auto-correct", "Watch live progress",
        "View the page with overlays", "Review disputed words", "Edit lines, undo auto-fixes",
        "Export Markdown / TXT / JSON", "Compare engines (CER / WER)", "View research results",
    ]
    for i, text in enumerate(cases):
        y = 0.85 - i * 0.097
        ex = 0.67
        ax.plot([cx + 0.05, ex - 0.2], [cy + 0.06, y], color=LINE, lw=0.9, zorder=1)
        ax.add_patch(matplotlib.patches.Ellipse((ex, y), 0.4, 0.075, facecolor="#ffffff", edgecolor=ACCENT, lw=1.1, zorder=2))
        ax.text(ex, y, text, ha="center", va="center", fontsize=8, color=INK, zorder=3)
    save(fig, "fig_usecase")


def sequence_diagram():
    fig, ax = canvas(7.2, 4.6)
    actors = [("Browser (Studio)", 0.12), ("FastAPI /api/v1/jobs", 0.4), ("Job worker", 0.66), ("OCR engine(s)", 0.89)]
    for name, x in actors:
        box(ax, x - 0.09, 0.89, 0.18, 0.08, name, size=8.6)
        ax.plot([x, x], [0.04, 0.89], color=LINE, lw=1.0, ls=(0, (4, 3)))
    steps = [
        (0.12, 0.4, 0.82, "POST file + engine + options"),
        (0.4, 0.66, 0.76, "start thread"),
        (0.4, 0.12, 0.7, "202 Accepted {job id}"),
        (0.66, 0.89, 0.64, "detect lines, read crops"),
        (0.89, 0.66, 0.58, "report(stage, lines, member)"),
        (0.12, 0.4, 0.5, "GET /jobs/{id}  (every 350 ms)"),
        (0.4, 0.12, 0.44, "snapshot: stage, lines 5/8, models"),
        (0.66, 0.4, 0.36, "result (blocks, words, timings)"),
        (0.12, 0.4, 0.28, "GET /jobs/{id}"),
        (0.4, 0.12, 0.22, "status = done + OCR result"),
        (0.12, 0.4, 0.12, "DELETE /jobs/{id}  (only if cancelled)"),
    ]
    for x1, x2, y, label in steps:
        dashed = x2 < x1
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=10,
                                     color=ACCENT if not dashed else MUTED, lw=1.1,
                                     linestyle="--" if dashed else "-"))
        ax.text((x1 + x2) / 2, y + 0.018, label, ha="center", fontsize=7.6, color=INK)
    save(fig, "fig_sequence")


def model_architectures():
    fig, ax = canvas(7.4, 6.4)
    cols = [
        ("CNN-BiLSTM (baseline)", COLORS["cnn_bilstm"], [
            ("Input line", "1 × 32 × 512 grey"),
            ("Conv 3×3, 32 + BN + ReLU", "pool 2×2 → 32 × 16 × 256"),
            ("Conv 3×3, 64 + BN + ReLU", "pool 2×2 → 64 × 8 × 128"),
            ("Conv 3×3, 128 + BN + ReLU", "pool 2×1 → 128 × 4 × 128"),
            ("Conv 3×3, 256 + BN + ReLU", "pool 2×1 → 256 × 2 × 128"),
            ("Map to sequence", "128 steps × 512 features"),
            ("2-layer BiLSTM, 192", "128 × 384, dropout 0.25"),
            ("Linear + softmax", "128 × 83 classes"),
            ("CTC loss / greedy decode", "2.39 M parameters"),
        ]),
        ("CRNN (Shi et al., 2016)", COLORS["crnn"], [
            ("Input line", "1 × 32 × 512 grey"),
            ("Conv 32, pool 2×2", "32 × 16 × 256"),
            ("Conv 64, pool 2×2", "64 × 8 × 128"),
            ("Conv 128 → Conv 128", "pool 2×1 → 128 × 4 × 128"),
            ("Conv 256 → Conv 256", "pool 2×1 → 256 × 2 × 128"),
            ("Conv 2×2, 256 (no pad)", "256 × 1 × 127"),
            ("BiLSTM 192 + linear", "127 × 192"),
            ("BiLSTM 192 + linear", "127 × 83 classes"),
            ("CTC loss / greedy decode", "2.78 M parameters"),
        ]),
        ("TrOCR-small (transfer learning)", COLORS["trocr_ft"], [
            ("Input line", "3 × 384 × 384 RGB"),
            ("Split into 16 × 16 patches", "576 patch tokens"),
            ("Linear patch embedding", "+ position embeddings"),
            ("DeiT encoder", "12 self-attention layers"),
            ("Encoder states", "visual features of the line"),
            ("Transformer decoder", "6 layers, cross-attention"),
            ("Autoregressive decoding", "one word-piece at a time"),
            ("Word-piece vocabulary", "→ detokenised text"),
            ("Fine-tuned 300 steps", "61.6 M parameters"),
        ]),
    ]
    width = 0.3
    for c, (title, color, layers) in enumerate(cols):
        x = 0.025 + c * 0.33
        ax.text(x + width / 2, 0.965, title, ha="center", fontsize=9.6, fontweight="bold", color=color)
        n = len(layers)
        top, bottom = 0.92, 0.03
        step = (top - bottom) / n
        for i, (name, shape) in enumerate(layers):
            h = step * 0.66
            y = top - (i + 1) * step + (step - h) / 2
            edge = color if i in (0, n - 1) else LINE
            box(ax, x, y, width, h, name, shape, fill="#ffffff", edge=edge, size=8.0, sub_size=7.0)
            if i < n - 1:
                gap_top = y - 0.012
                gap_bottom = y - (step - h) + 0.012
                arrow(ax, x + width / 2, gap_top, x + width / 2, gap_bottom, lw=1.0)
    save(fig, "fig_models")


def rover_example():
    fig, ax = canvas(7.2, 3.6)
    rows = [("TrOCR (fine-tuned)", "trocr_ft", 3, "fan"), ("CRNN", "crnn", 1, "fan"), ("TrOCR (pretrained)", "trocr", 3, "Pan"),
            ("PaddleOCR", "paddle", 2, "Pan"), ("CNN-BiLSTM", "cnn_bilstm", 2, "Pan")]
    ax.text(0.02, 0.93, "Five readings of the same words, aligned into slots", fontsize=9.5, fontweight="bold", color=INK)
    headers = ["Model (weight)", "slot 1", "slot 2", "slot 3"]
    xs = [0.02, 0.28, 0.4, 0.52]
    for x, head in zip(xs, headers):
        ax.text(x, 0.84, head, fontsize=8.2, color=MUTED, fontweight="bold")
    for i, (name, key, weight, word) in enumerate(rows):
        y = 0.75 - i * 0.11
        ax.add_patch(plt.Circle((xs[0] + 0.008, y + 0.012), 0.008, color=COLORS[key]))
        ax.text(xs[0] + 0.025, y, f"{name} ({weight})", fontsize=8.4, color=INK)
        ax.text(xs[1], y, "Tin", fontsize=9, color=INK)
        ax.text(xs[2], y, word, fontsize=9, color=ACCENT if word == "Pan" else "#b42318", fontweight="bold")
        ax.text(xs[3], y, "Alley", fontsize=9, color=INK)
    ax.add_patch(FancyBboxPatch((xs[2] - 0.015, 0.27), 0.085, 0.6, boxstyle="round,pad=0.005,rounding_size=0.01",
                                linewidth=1.3, edgecolor=ACCENT, facecolor="none"))
    ax.text(0.02, 0.06, "Vote for a word = model weight × that model's confidence on the line. Words without a majority are flagged.",
            fontsize=7.8, color=MUTED, style="italic")
    # Vote bars.
    bx = fig.add_axes([0.7, 0.3, 0.27, 0.5])
    bx.barh(["fan", "Pan"], [34, 66], color=["#fda29b", ACCENT], height=0.55)
    bx.set_xlim(0, 100)
    bx.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    bx.set_title("Slot 2: share of the vote", fontsize=9)
    for y, v in enumerate([34, 66]):
        bx.text(v + 2, y, f"{v}%", va="center", fontsize=9, color=INK, fontweight="bold")
    bx.grid(axis="y", visible=False)
    ax.text(0.835, 0.12, "Result: “Tin Pan Alley” (correct)", ha="center", fontsize=8.6, color=ACCENT, fontweight="bold")
    save(fig, "fig_rover")


# --------------------------------------------------------------------------- data figures
def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def dataset_samples():
    rows = []
    for name in ("iam", "synthetic"):
        base = ROOT / "data" / "processed" / name
        lines = [l.split("\t", 1) for l in (base / "test.txt").read_text(encoding="utf-8").splitlines() if "\t" in l]
        picks = [lines[i] for i in ((0, 2, 3) if name == "iam" else (1, 6, 0))]
        for rel, text in picks:
            rows.append((name, read_gray(base / rel), text))
    # Pad every line to the same aspect ratio (left-aligned) so none is stretched.
    ratio = max(im.shape[1] / im.shape[0] for _, im, _ in rows)
    padded = []
    for name, im, text in rows:
        h, w = im.shape
        fill = int(np.median(np.concatenate([im[:, -3:].ravel(), im[:3].ravel()])))
        canvas_img = np.full((h, int(round(h * ratio))), fill, dtype=np.uint8)
        canvas_img[:, :w] = im
        padded.append((name, canvas_img, text))
    rows = padded
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, len(rows) * (6.9 / ratio + 0.34)))
    from datasets.line_image import normalize_transcription
    for ax, (name, img, text) in zip(axes, rows):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(LINE)
        tag = "IAM handwriting" if name == "iam" else "Synthetic"
        ax.set_title(f"{tag}:  “{normalize_transcription(text)}”", fontsize=8.6, loc="left", fontweight="normal", color=INK)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig_datasets")


def augmentation_examples():
    from datasets.augmentations import OCRAugmentor
    base = ROOT / "data" / "processed" / "iam"
    rel = [l.split("\t", 1)[0] for l in (base / "train.txt").read_text(encoding="utf-8").splitlines() if "\t" in l][5]
    img = read_gray(base / rel)
    aug = OCRAugmentor({"enabled": True, "rotation_degrees": 1.5, "brightness_range": [0.6, 1.4], "contrast_range": [0.6, 1.4],
                        "noise_std": 0.04, "blur_probability": 0.3, "blur_kernel_max": 2, "scale_range": [0.8, 1.15],
                        "perspective_distortion": 0.08})
    rng_state = np.random.get_state()
    np.random.seed(4)
    variants = [("Original", img)]
    ops = [("Rotation", aug._random_rotate), ("Scaling", aug._random_scale), ("Gaussian noise", aug._add_gaussian_noise),
           ("Blur", aug._random_blur), ("Brightness / contrast", aug._random_brightness_contrast), ("Perspective", aug._random_perspective)]
    for name, fn in ops:
        out = img.copy()
        for _ in range(3 if name == "Gaussian noise" else 1):
            out = fn(out)
        variants.append((name, out))
    np.random.set_state(rng_state)
    fig, axes = plt.subplots(len(variants), 1, figsize=(7.0, 6.2))
    for ax, (name, im) in zip(axes, variants):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.6, color=INK)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(LINE)
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig_augmentation")


def preprocessing_example():
    from api.preprocess import preprocess_page
    data = np.fromfile(str(ROOT / "samples" / "noisy_scan.jpg"), dtype=np.uint8)
    page = cv2.imdecode(data, cv2.IMREAD_COLOR)
    out, report = preprocess_page(page, ["deskew", "denoise", "contrast"], with_previews=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, im, title in ((axes[0], page, "Original: skewed, noisy, uneven lighting"),
                          (axes[1], out, f"After deskew ({report['deskew_angle']:+.1f}°), denoise and CLAHE")):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8.8, fontweight="normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, "fig_preprocessing")


# --------------------------------------------------------------------------- charts
def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for key in ("cnn_bilstm", "crnn"):
        run = load(TRAIN / f"{key}_history.json")
        h = run["history"]
        epochs = np.arange(1, len(h["val_cer"]) + 1)
        axes[0].plot(epochs, np.array(h["val_cer"]) * 100, marker="o", ms=3.5, color=COLORS[key],
                     label=f"{LABELS[key]} (best {run['best_val_cer'] * 100:.1f}%)")
        axes[1].plot(epochs, h["train_loss"], marker="o", ms=3.5, color=COLORS[key], label=f"{LABELS[key]} train")
        axes[1].plot(epochs, h["val_loss"], ls="--", color=COLORS[key], alpha=0.8, label=f"{LABELS[key]} validation")
    axes[0].set_title("Validation CER per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CER (%)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("CTC loss per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7.5, ncol=1)
    fig.tight_layout()
    save(fig, "fig_training")


def finetune_curve():
    run = load(TRAIN / "trocr_ft_history.json")
    h = run["history"]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(h["step"], h["val_cer_synthetic"], marker="o", color=COLORS["trocr_ft"], label="Printed & degraded (synthetic)")
    ax.plot(h["step"], h["val_cer_iam"], marker="o", color=COLORS["trocr"], label="IAM handwriting")
    ax.set_xlabel("Fine-tuning step (batch 8)")
    ax.set_ylabel("Validation CER (%)")
    ax.set_title("TrOCR fine-tuning")
    for x, y in zip(h["step"], h["val_cer_synthetic"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(9, -12) if x else (10, -3), ha="center",
                    fontsize=7.5, color=COLORS["trocr_ft"])
    ax.set_ylim(0, 25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_finetune")


def comparison():
    summary = load(EVAL / "summary.json")["models"]
    order = ["ensemble", "trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (set_id, title) in zip(axes, (("iam", "IAM handwriting"), ("synthetic", "Printed & degraded"))):
        cer = [summary[m][set_id]["cer"] for m in order]
        wer = [summary[m][set_id]["wer"] for m in order]
        y = np.arange(len(order))
        ax.barh(y - 0.19, cer, height=0.36, color=[COLORS[m] for m in order], label="CER")
        ax.barh(y + 0.19, wer, height=0.36, color=[COLORS[m] for m in order], alpha=0.35, label="WER")
        for i, (c, w) in enumerate(zip(cer, wer)):
            ax.text(c + 0.8, i - 0.19, f"{c:.1f}", va="center", fontsize=7.6, color=INK)
            ax.text(w + 0.8, i + 0.19, f"{w:.1f}", va="center", fontsize=7.2, color=MUTED)
        ax.set_yticks(y, [LABELS[m] for m in order])
        ax.set_title(title)
        ax.set_xlabel("Error rate (%), lower is better")
        ax.set_xlim(0, max(wer) * 1.15)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # shared y axis: invert once so the best model is on top
    axes[1].legend(["CER (solid)", "WER (light)"], fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_comparison")


def ensemble_chart():
    ens = load(EVAL / "ensemble.json")
    members = list(ens["weights"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (set_id, title) in zip(axes, (("iam", "IAM handwriting"), ("synthetic", "Printed & degraded"))):
        rows = ens["sets"][set_id]
        best = min((rows[m] for m in members), key=lambda r: r["cer"])
        groups = [("Best single model", best, "#98a2b3"), ("Consensus vote", rows["ensemble_cv"], ACCENT),
                  ("Oracle ceiling", rows["oracle_best_line"], "#d0d5dd")]
        x = np.arange(2)
        for i, (label, r, color) in enumerate(groups):
            vals = [r["cer"], r["wer"]]
            bars = ax.bar(x + (i - 1) * 0.26, vals, width=0.25, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7.4, color=INK)
        ax.set_xticks(x, ["CER (%)", "WER (%)"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_ensemble")


def robustness():
    data = load(EVAL / "robustness.json")
    order = ["cnn_bilstm", "crnn", "trocr", "trocr_ft", "paddle"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
    for ax, cond in zip(axes.flat, data["conditions"]):
        xs = np.arange(len(cond["levels"]))
        for m in order:
            ax.plot(xs, data["models"][m][cond["id"]], marker="o", ms=3, color=COLORS[m], label=LABELS[m])
        ax.set_xticks(xs, [str(v) for v in cond["levels"]])
        ax.set_title(cond["label"], fontsize=9.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("CER (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_robustness")


def speed_accuracy():
    summary = load(EVAL / "summary.json")["models"]
    order = ["ensemble", "trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    offsets = {"crnn": (6, -10), "cnn_bilstm": (6, 4), "trocr": (-9, 6), "trocr_ft": (8, 8), "ensemble": (8, -13), "paddle": (6, 2)}
    for m in order:
        ms = np.mean([summary[m][s]["ms_per_line"] for s in ("iam", "synthetic")])
        cer = summary[m]["iam"]["cer"]
        ax.scatter(ms, cer, s=70, color=COLORS[m], zorder=3)
        dx, dy = offsets[m]
        ax.annotate(LABELS[m], (ms, cer), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    ha="right" if dx < 0 else "left", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Time per line on a laptop CPU (ms, log scale)")
    ax.set_ylabel("Handwriting CER (%)")
    ax.set_title("Accuracy versus speed")
    ax.set_xlim(7, 3000)
    ax.set_ylim(2.5, 31)
    fig.tight_layout()
    save(fig, "fig_speed")


if __name__ == "__main__":
    system_architecture()
    deployment_architecture()
    use_case()
    sequence_diagram()
    model_architectures()
    rover_example()
    dataset_samples()
    augmentation_examples()
    preprocessing_example()
    training_curves()
    finetune_curve()
    comparison()
    ensemble_chart()
    robustness()
    speed_accuracy()
