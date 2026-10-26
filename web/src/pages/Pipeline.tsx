import { motion } from "framer-motion";
import { ArrowRight, FileInput, LayoutPanelTop, ScanSearch, SpellCheck2, Type, Wand2, FileOutput } from "lucide-react";

const STAGES = [
  {
    icon: FileInput,
    title: "Input document",
    body: "PDF pages are rendered with PyMuPDF; images (PNG/JPG/TIFF/WEBP, multi-frame) are decoded with EXIF orientation. Size, page and pixel limits stop decompression bombs.",
    code: "api/documents.py",
  },
  {
    icon: Wand2,
    title: "Image preprocessing",
    body: "Projection-profile deskew (±12°), Gaussian + median denoising, CLAHE contrast, adaptive binarisation. Each line crop is then polarity- and brightness-normalised exactly as during training.",
    code: "api/preprocess.py · preprocessing/ · datasets/line_image.py",
  },
  {
    icon: ScanSearch,
    title: "Text detection",
    body: "A DBNet detector (PP-OCR) finds text-line quadrilaterals, which are perspective-rectified with a context margin. An OpenCV MSER detector is the fallback.",
    code: "api/engines.py · detection/opencv_detector.py",
  },
  {
    icon: LayoutPanelTop,
    title: "Layout analysis",
    body: "Lines are grouped into reading order. Titles and headings are found by size, tables by ruled grids or aligned cells, checkboxes by contour analysis, and signature fields by their labels.",
    code: "api/structure.py · layout/layout_analyzer.py",
  },
  {
    icon: Type,
    title: "Deep-learning recognition",
    body: "Choose an engine: our CNN-BiLSTM baseline or CRNN (both CTC, trained here on IAM + synthetic data), TrOCR (ViT + Transformer, transfer learning), or the PaddleOCR reference.",
    code: "models/ · models/recognizer.py · training/",
  },
  {
    icon: SpellCheck2,
    title: "Post-processing",
    body: "Context-aware digit/letter fixes (2O24 → 2024), case-preserving spell correction that skips names and acronyms, and confidence-based [Illegible] flags. Every change is reported.",
    code: "postprocessing/text_corrector.py",
  },
  {
    icon: FileOutput,
    title: "Extracted digital text",
    body: "Editable Markdown with tables and checkboxes, a review queue and confidence heatmaps, exported as .md, .txt or JSON.",
    code: "web/src/",
  },
];

export function Pipeline({ onTry }: { onTry: () => void }) {
  return (
    <div className="pipeline-page">
      <header className="page-head">
        <p className="eyebrow"><span className="pulse-dot" /> System architecture</p>
        <h1>Seven stages from <span className="gradient-text">pixels to text.</span></h1>
        <p className="hero-sub">The end-to-end pipeline from the project proposal, implemented and wired into the web app. Each stage maps to a module in the repository.</p>
      </header>
      <ol className="arch-timeline">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon;
          return (
            <motion.li
              key={stage.title}
              className="glass timeline-card"
              initial={{ opacity: 0, x: index % 2 ? 30 : -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            >
              <span className="timeline-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="timeline-icon"><Icon size={20} /></span>
              <div>
                <h3>{stage.title}</h3>
                <p>{stage.body}</p>
                <code>{stage.code}</code>
              </div>
            </motion.li>
          );
        })}
      </ol>
      <div className="cta-row">
        <button type="button" className="primary-button big" onClick={onTry}>Try it on a document <ArrowRight size={18} /></button>
      </div>
    </div>
  );
}
