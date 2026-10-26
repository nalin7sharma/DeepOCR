#!/usr/bin/env python
"""
DeepOCR — Streamlit Web Application
====================================
Professional, modular OCR interface allowing users to:
  - Upload image or PDF documents
  - Preview original and preprocessed images
  - View detected text regions with bounding boxes
  - Select OCR model
  - Extract and display text
  - Download results
  - View inference timing
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepOCR — Intelligent Document Recognition",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .hero p  { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        color: white;
        margin: 0.25rem;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

    /* Region type badge */
    .badge-title     { background:#7c3aed; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
    .badge-heading   { background:#1d4ed8; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
    .badge-paragraph { background:#065f46; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; }

    /* Warning box */
    .warn-box {
        background: #1c1917;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #fef3c7;
    }

    /* Result text area */
    .stTextArea textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem;
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    /* Step header */
    .step-header {
        font-size: 1rem;
        font-weight: 600;
        color: #38bdf8;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.5rem;
        margin: 1rem 0 0.5rem;
    }

    /* Hide footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📄 DeepOCR</h1>
    <p>Intelligent Deep Learning-Based OCR for Handwritten & Complex Documents</p>
</div>
""", unsafe_allow_html=True)


# ── Lazy imports with error handling ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_pipeline():
    """Import the OCR pipeline (cached so it only loads once)."""
    try:
        from inference.ocr_pipeline import OCRPipeline
        return OCRPipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def _load_pipeline(model_name: str):
    """Build (once per model) the trained-model OCR pipeline."""
    from inference.ocr_pipeline import OCRPipeline
    return OCRPipeline.from_config(model_name=model_name)


@st.cache_resource(show_spinner=False)
def _import_processor():
    try:
        from preprocessing.image_processor import ImageProcessor
        return ImageProcessor, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def _import_detector():
    try:
        from detection.opencv_detector import OpenCVTextDetector
        return OpenCVTextDetector, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def _import_layout():
    try:
        from layout.layout_analyzer import LayoutAnalyzer
        return LayoutAnalyzer, None
    except Exception as e:
        return None, str(e)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 OCR Model",
        ["cnn_bilstm", "crnn", "transformer_ocr"],
        format_func=lambda x: {
            "cnn_bilstm": "CNN + BiLSTM (Baseline)",
            "crnn": "CRNN (Shi et al.)",
            "transformer_ocr": "TrOCR Transformer (transfer learning)",
        }[x],
        help="Select the deep learning model for text recognition",
    )

    st.markdown("---")
    st.markdown("### 🔧 Preprocessing")
    do_deskew    = st.checkbox("Deskew", value=True)
    do_denoise   = st.checkbox("Denoise", value=True)
    do_contrast  = st.checkbox("CLAHE Contrast", value=True)
    do_normalize = st.checkbox("Normalize brightness", value=True)
    do_sharpen   = st.checkbox("Sharpen", value=False)

    st.markdown("---")
    st.markdown("### 🔍 Detection")
    detect_method = st.radio(
        "Detector",
        ["MSER (default)", "Morphological"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📝 Post-Processing")
    do_spell = st.checkbox("Spell correction", value=True)
    do_char_subs = st.checkbox("Character substitutions", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>DeepOCR v1.0 | Academic Project</small>",
        unsafe_allow_html=True,
    )


# ── File uploader ──────────────────────────────────────────────────────────────
st.markdown('<div class="step-header">① Upload Document</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload an image or PDF",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "pdf"],
    label_visibility="collapsed",
)


def load_image_from_upload(file) -> list[np.ndarray]:
    """Convert uploaded file to list of BGR numpy arrays (one per page for PDFs)."""
    suffix = Path(file.name).suffix.lower()
    images = []

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                mat = fitz.Matrix(2, 2)  # 2x zoom for better resolution
                pix = page.get_pixmap(matrix=mat)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
        except ImportError:
            st.warning("PDF support requires PyMuPDF (`pip install pymupdf`). "
                       "Processing first page as image instead.")
            file.seek(0)
            img_pil = Image.open(file).convert("RGB")
            images.append(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return []
    else:
        img_pil = Image.open(file).convert("RGB")
        images.append(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

    return images


# ── Main processing area ───────────────────────────────────────────────────────
if uploaded is not None:
    images = load_image_from_upload(uploaded)

    if not images:
        st.error("Could not load the uploaded file.")
        st.stop()

    # Multi-page selector for PDFs
    page_idx = 0
    if len(images) > 1:
        page_idx = st.slider(f"Page ({len(images)} pages)", 1, len(images), 1) - 1

    bgr_image = images[page_idx]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # ── Row 1: Original + Preprocessed ────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="step-header">② Original Document</div>', unsafe_allow_html=True)
        st.image(rgb_image, caption=f"Original: {uploaded.name}", use_container_width=True)

    with col2:
        st.markdown('<div class="step-header">③ Preprocessed Image</div>', unsafe_allow_html=True)
        ImageProcessorCls, proc_err = _import_processor()
        if ImageProcessorCls and not proc_err:
            proc_config = {
                "deskew": do_deskew,
                "denoise": do_denoise,
                "contrast_enhance": do_contrast,
                "normalize": do_normalize,
                "sharpen": do_sharpen,
                "resize": False,   # Don't resize for display
            }
            processor = ImageProcessorCls(proc_config)
            preprocessed_np = processor.process(bgr_image)
            if len(preprocessed_np.shape) == 2:
                st.image(preprocessed_np, caption="Preprocessed (Grayscale)", use_container_width=True)
            else:
                st.image(cv2.cvtColor(preprocessed_np, cv2.COLOR_BGR2RGB),
                         caption="Preprocessed", use_container_width=True)
        else:
            st.warning(f"Preprocessor unavailable: {proc_err}")
            preprocessed_np = bgr_image

    # ── Row 2: Detection + Layout ──────────────────────────────────────────
    st.markdown('<div class="step-header">④ Text Detection & Layout Analysis</div>', unsafe_allow_html=True)

    DetectorCls, det_err = _import_detector()
    LayoutCls, lay_err = _import_layout()

    if DetectorCls and LayoutCls:
        with st.spinner("Detecting text regions..."):
            detector = DetectorCls()
            if detect_method == "Morphological":
                detections = detector.detect_simple(bgr_image)
            else:
                detections = detector.detect(bgr_image)

            layout_analyzer = LayoutCls()
            h_img, w_img = bgr_image.shape[:2]
            layout_blocks = layout_analyzer.analyze(detections, w_img, h_img)

        # Draw bboxes
        annotated = bgr_image.copy()
        type_colors = {
            "title":     (128, 0, 255),    # purple
            "heading":   (255, 128, 0),    # orange
            "paragraph": (0, 200, 80),     # green
            "unknown":   (100, 100, 100),  # grey
        }
        for block in layout_blocks:
            x, y, bw, bh = block.bbox
            color = type_colors.get(block.region_type, (0, 200, 80))
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            label = f"{block.reading_order+1}:{block.region_type[:3]}"
            cv2.putText(annotated, label, (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        col3, col4 = st.columns([2, 1])
        with col3:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"Detected regions: {len(layout_blocks)}", use_container_width=True)
        with col4:
            st.markdown("**Region Summary**")
            from collections import Counter
            type_counts = Counter(b.region_type for b in layout_blocks)
            for rtype, cnt in type_counts.items():
                badge_cls = f"badge-{rtype}" if rtype in ("title","heading","paragraph") else "badge-paragraph"
                st.markdown(
                    f'<span class="{badge_cls}">{rtype}</span> &nbsp; **{cnt}** region(s)',
                    unsafe_allow_html=True
                )
            st.markdown(f"\n**Total regions:** {len(layout_blocks)}")
    else:
        st.warning(f"Detector/Layout unavailable: {det_err or lay_err}")
        layout_blocks = []

    # ── Row 3: OCR Recognition ─────────────────────────────────────────────
    st.markdown('<div class="step-header">⑤ OCR Recognition</div>', unsafe_allow_html=True)

    run_ocr = st.button("🚀 Extract Text", type="primary", use_container_width=True)

    if run_ocr:
        OCRPipelineCls, pip_err = _import_pipeline()

        if pip_err:
            st.error(f"Pipeline import error: {pip_err}")
            st.stop()

        with st.spinner(f"Running {model_choice} OCR pipeline..."):
            t0 = time.perf_counter()
            try:
                pipeline = _load_pipeline(model_choice)

                # Run full pipeline
                result = pipeline.run(bgr_image, correct=do_spell or do_char_subs)
                elapsed_ms = (time.perf_counter() - t0) * 1000

            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"OCR Error: {e}")
                st.exception(e)
                st.stop()

        # ── Metrics ────────────────────────────────────────────────────────
        st.markdown("**Inference Summary**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{result.get("inference_ms", elapsed_ms):.0f}</div>
                <div class="label">ms total</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(result.get("regions", []))}</div>
                <div class="label">regions</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            words = len(result.get("text","").split())
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{words}</div>
                <div class="label">words</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            chars = len(result.get("text",""))
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{chars}</div>
                <div class="label">characters</div>
            </div>""", unsafe_allow_html=True)

        # ── Annotated bbox image ───────────────────────────────────────────
        if result.get("bbox_image") is not None:
            bbox_rgb = cv2.cvtColor(result["bbox_image"], cv2.COLOR_BGR2RGB)
            st.image(bbox_rgb, caption="OCR-recognised regions", use_container_width=True)

        # ── Text output ────────────────────────────────────────────────────
        st.markdown('<div class="step-header">⑥ Extracted Text</div>', unsafe_allow_html=True)

        full_text = result.get("text", "")
        st.text_area(
            "Recognised text (copy from here)",
            value=full_text if full_text.strip() else "(no text recognised — train the model first)",
            height=300,
            label_visibility="collapsed",
        )

        # Per-region breakdown
        with st.expander("📋 Per-Region Results", expanded=False):
            for i, reg in enumerate(result.get("regions", [])):
                x, y, bw, bh = reg["bbox"]
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**Region {i+1}** · `{reg['region_type']}`<br>"
                                f"<small>({x},{y}) {bw}×{bh}</small>",
                                unsafe_allow_html=True)
                with col_b:
                    st.text(reg.get("text", "") or "(empty)")
                st.divider()

        # ── Download buttons ───────────────────────────────────────────────
        st.markdown('<div class="step-header">⑦ Download Results</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                label="⬇️ Download as .txt",
                data=full_text.encode("utf-8"),
                file_name=f"{Path(uploaded.name).stem}_ocr.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            # Download annotated image
            if result.get("bbox_image") is not None:
                success, buf = cv2.imencode(".png", result["bbox_image"])
                if success:
                    st.download_button(
                        label="⬇️ Download annotated image",
                        data=buf.tobytes(),
                        file_name=f"{Path(uploaded.name).stem}_regions.png",
                        mime="image/png",
                        use_container_width=True,
                    )

else:
    # Landing state when no file is uploaded
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#64748b;">
        <div style="font-size:4rem;">📂</div>
        <h3 style="color:#94a3b8;">Upload a document to begin</h3>
        <p>Supported formats: PNG, JPG, BMP, TIFF, PDF</p>
        <p>Handwritten or printed text — complex layouts welcome</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Info about models ──────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 🔷 CNN + BiLSTM
        **Baseline model.** CNN feature extractor combined with bidirectional
        LSTM for sequence modelling and CTC loss for end-to-end training.
        Fast and lightweight.
        """)
    with c2:
        st.markdown("""
        #### 🔶 CRNN
        **Shi et al. (2016)** VGG-inspired CNN backbone with deeper feature
        extraction, paired with two-layer BiLSTM. Proven architecture for
        scene text recognition.
        """)
    with c3:
        st.markdown("""
        #### 🌟 Transformer OCR
        **Advanced model.** Vision Transformer encoder with autoregressive
        decoder. Wraps Microsoft TrOCR (pretrained) or custom ViT
        for fine-tuning.
        """)

    st.info("💡 **Quick Start:** Generate synthetic training data with `python scripts/prepare_dataset.py --dataset synthetic`, then train with `python scripts/train.py --model cnn_bilstm`")
