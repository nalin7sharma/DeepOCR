import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Flame, Layers, Maximize2, SplitSquareHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import type { OcrPage, StructuredBlock } from "../types";
import { confidenceColor } from "../lib/text";

export const KIND_COLORS: Record<string, string> = {
  text: "#22d3ee",
  title: "#f472b6",
  heading: "#a78bfa",
  table: "#a3e635",
  checkbox: "#fbbf24",
  signature: "#fb7185",
  illegible: "#f87171",
};

export const KIND_LABELS: Record<string, string> = {
  text: "Text",
  title: "Title",
  heading: "Heading",
  table: "Table",
  checkbox: "Checkbox",
  signature: "Signature field",
  illegible: "Unclear text",
};

type Layer = "structure" | "heat" | "clean";

interface Props {
  pages: OcrPage[];
  pageIndex: number;
  onPageChange: (index: number) => void;
  selected: number | null;
  onSelect: (index: number | null, block?: StructuredBlock) => void;
}

const points = (polygon: number[][]) => polygon.map((point) => point.join(",")).join(" ");

export function DocumentViewer({ pages, pageIndex, onPageChange, selected, onSelect }: Props) {
  const page = pages[pageIndex];
  const [layer, setLayer] = useState<Layer>("structure");
  const [zoom, setZoom] = useState(1);
  const [stage, setStage] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [split, setSplit] = useState(50);
  const [hover, setHover] = useState<{ x: number; y: number; title: string; text: string; confidence: number } | null>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  const stages = page?.preprocessing?.stages ?? [];
  const original = stages.find((item) => item.id === "original");
  useEffect(() => {
    setStage(null);
    setCompare(false);
  }, [pageIndex, page?.preview_image]);

  const stageImage = stage ? stages.find((item) => item.id === stage)?.image : null;
  const mainImage = `data:image/jpeg;base64,${stageImage ?? page?.preview_image}`;
  const showOverlay = !stage && !compare && layer !== "clean";

  const regionsByConfidence = useMemo(
    () => [...(page?.regions ?? [])].sort((a, b) => b.confidence - a.confidence),
    [page],
  );

  if (!page) return null;

  const showTip = (event: React.MouseEvent, title: string, text: string, confidence: number) => {
    const bounds = frameRef.current?.getBoundingClientRect();
    if (!bounds) return;
    setHover({ x: event.clientX - bounds.left, y: event.clientY - bounds.top, title, text, confidence });
  };

  return (
    <div className="viewer">
      <div className="viewer-toolbar">
        <div className="segmented" role="radiogroup" aria-label="Overlay">
          {([
            ["structure", Layers, "Structure"],
            ["heat", Flame, "Confidence"],
            ["clean", Maximize2, "Clean"],
          ] as const).map(([id, Icon, label]) => (
            <button key={id} type="button" role="radio" aria-checked={layer === id} className={layer === id ? "on" : ""} onClick={() => setLayer(id)}>
              <Icon size={14} /> {label}
            </button>
          ))}
        </div>
        <div className="viewer-tools">
          {original && (
            <button type="button" className={`tool ${compare ? "on" : ""}`} onClick={() => { setCompare(!compare); setStage(null); }} title="Before / after preprocessing">
              <SplitSquareHorizontal size={15} /> Before/after
            </button>
          )}
          <button type="button" className="tool" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.25).toFixed(2)))}><ZoomOut size={15} /></button>
          <button type="button" className="tool zoom-label" onClick={() => setZoom(1)} title="Fit">{Math.round(zoom * 100)}%</button>
          <button type="button" className="tool" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(4, +(z + 0.25).toFixed(2)))}><ZoomIn size={15} /></button>
        </div>
      </div>

      <div className="viewer-stage" ref={frameRef} onMouseLeave={() => setHover(null)}>
        <div className="viewer-canvas" style={{ width: `${zoom * 100}%` }}>
          {compare && original ? (
            <div className="compare-wrap">
              <img className="doc-image" src={`data:image/jpeg;base64,${original.image}`} alt={`Original page ${page.page_number}`} />
              <img
                className="doc-image compare-top"
                src={`data:image/jpeg;base64,${page.preview_image}`}
                alt={`Processed page ${page.page_number}`}
                style={{ clipPath: `inset(0 0 0 ${split}%)` }}
              />
              <span className="compare-line" style={{ left: `${split}%` }}><i /></span>
              <input
                className="compare-range"
                type="range"
                min={0}
                max={100}
                value={split}
                aria-label="Before / after position"
                onChange={(event) => setSplit(Number(event.target.value))}
              />
              <span className="compare-tag left">Original</span>
              <span className="compare-tag right">Processed</span>
            </div>
          ) : (
            <img className="doc-image" src={mainImage} alt={`Document page ${page.page_number}`} />
          )}
          <span key={page.preview_image.slice(-24)} className="scan-sweep" aria-hidden="true" />
          {showOverlay && (
            <svg className="overlay" viewBox={`0 0 ${page.width} ${page.height}`} preserveAspectRatio="none" role="group" aria-label="Recognised regions">
              {layer === "structure" && page.blocks.map((block, index) => (
                <motion.polygon
                  key={`b-${index}`}
                  points={points(block.polygon)}
                  className={`block-shape ${selected === index ? "selected" : ""} ${block.review_required ? "review" : ""}`}
                  style={{ ["--kind" as string]: KIND_COLORS[block.kind] ?? "#22d3ee" }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: Math.min(1.2, index * 0.025) }}
                  role="button"
                  tabIndex={0}
                  aria-label={`${KIND_LABELS[block.kind] ?? block.kind} ${index + 1}: ${block.text}`}
                  onClick={() => onSelect(selected === index ? null : index, block)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect(index, block);
                    }
                  }}
                  onMouseMove={(event) => showTip(event, KIND_LABELS[block.kind] ?? block.kind, block.raw_text || block.text, block.confidence)}
                />
              ))}
              {layer === "heat" && regionsByConfidence.map((region, index) => (
                <motion.polygon
                  key={`r-${index}`}
                  points={points(region.polygon)}
                  className="heat-shape"
                  fill={confidenceColor(region.confidence, 0.34)}
                  stroke={confidenceColor(region.confidence, 0.95)}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: Math.min(1, index * 0.015) }}
                  onMouseMove={(event) => showTip(event, "Recognised region", region.text, region.confidence)}
                />
              ))}
            </svg>
          )}
        </div>
        <AnimatePresence>
          {hover && (
            <motion.div
              className="viewer-tip"
              style={{ left: hover.x + 14, top: hover.y + 14 }}
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
            >
              <span className="tip-head">
                <b>{hover.title}</b>
                <em style={{ color: confidenceColor(hover.confidence) }}>{Math.round(hover.confidence * 100)}%</em>
              </span>
              <span className="tip-text">{hover.text || "—"}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {layer === "heat" && showOverlay && (
        <div className="heat-legend" aria-hidden="true">
          <span>low</span><i /><span>high confidence</span>
        </div>
      )}
      {layer === "structure" && showOverlay && (
        <div className="kind-legend">
          {Object.entries(KIND_LABELS).filter(([kind]) => page.blocks.some((b) => b.kind === kind)).map(([kind, label]) => (
            <span key={kind}><i style={{ background: KIND_COLORS[kind] }} />{label}</span>
          ))}
        </div>
      )}

      {stages.length > 1 && (
        <div className="stage-strip" aria-label="Preprocessing stages">
          {stages.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`stage-thumb ${(stage ?? stages[stages.length - 1].id) === item.id && !compare ? "on" : ""}`}
              onClick={() => { setCompare(false); setStage(item.id === stages[stages.length - 1].id ? null : item.id); }}
            >
              <img src={`data:image/jpeg;base64,${item.image}`} alt="" />
              <span>{item.label}</span>
              {item.ms > 0 && <small>{Math.round(item.ms)} ms</small>}
            </button>
          ))}
        </div>
      )}

      {pages.length > 1 && (
        <nav className="page-nav" aria-label="Document pages">
          <button type="button" className="tool" aria-label="Previous page" disabled={pageIndex === 0} onClick={() => onPageChange(pageIndex - 1)}><ChevronLeft size={16} /></button>
          <div className="page-strip" role="tablist" aria-label="Select a page">
            {pages.map((item, index) => (
              <button
                key={item.page_number}
                type="button"
                role="tab"
                aria-selected={index === pageIndex}
                className={`page-thumb ${index === pageIndex ? "on" : ""}`}
                onClick={() => onPageChange(index)}
              >
                <img src={`data:image/jpeg;base64,${item.preview_image}`} alt="" />
                <span>{String(item.page_number).padStart(2, "0")}</span>
              </button>
            ))}
          </div>
          <button type="button" className="tool" aria-label="Next page" disabled={pageIndex === pages.length - 1} onClick={() => onPageChange(pageIndex + 1)}><ChevronRight size={16} /></button>
        </nav>
      )}
    </div>
  );
}
