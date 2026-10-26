import { useEffect, useRef, useState } from "react";
import { EyeOff, Flame, GitCompareArrows, Layers, SplitSquareHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import type { OcrPage, StructuredBlock } from "../types";
import { agreement } from "../lib/document";
import { levelColor } from "../lib/text";
import { Segmented } from "./Ui";

export const KIND_COLORS: Record<string, string> = {
  text: "#2e90fa", heading: "#7a5af8", title: "#7a5af8", table: "#16b364", checkbox: "#f79009", signature: "#ee46bc", illegible: "#f04438",
};
export const KIND_LABELS: Record<string, string> = {
  text: "Text", heading: "Heading", title: "Title", table: "Table", checkbox: "Checkbox", signature: "Signature field", illegible: "Unclear text",
};

export type Overlay = "layout" | "agreement" | "confidence" | "off";

interface Props {
  page: OcrPage;
  blocks: StructuredBlock[];
  consensus: boolean;
  selected: number | null;
  onSelect: (blockIndex: number | null) => void;
}

const points = (polygon: number[][]) => polygon.map((p) => `${p[0]},${p[1]}`).join(" ");

export function DocumentViewer({ page, blocks, consensus, selected, onSelect }: Props) {
  const [overlay, setOverlay] = useState<Overlay>(consensus ? "agreement" : "layout");
  const [zoom, setZoom] = useState(1);
  const [split, setSplit] = useState<number | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; title: string; text: string; value: number | null; valueLabel: string } | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const original = page.preprocessing?.stages.find((stage) => stage.id === "original");

  useEffect(() => setOverlay(consensus ? "agreement" : "layout"), [consensus]);
  useEffect(() => setSplit(null), [page.page_number]);

  const showTip = (event: React.MouseEvent, title: string, text: string, value: number | null, valueLabel: string) => {
    const bounds = stageRef.current?.getBoundingClientRect();
    if (!bounds || !stageRef.current) return;
    setTip({ x: event.clientX - bounds.left + stageRef.current.scrollLeft + 14, y: event.clientY - bounds.top + stageRef.current.scrollTop + 14, title, text, value, valueLabel });
  };

  const overlayOptions: { value: Overlay; label: JSX.Element }[] = [
    { value: "layout", label: <><Layers size={14} />Layout</> },
    ...(consensus ? [{ value: "agreement" as Overlay, label: <><GitCompareArrows size={14} />Agreement</> }] : []),
    { value: "confidence", label: <><Flame size={14} />Confidence</> },
    { value: "off", label: <><EyeOff size={14} />Off</> },
  ];

  return (
    <section className="card docp" aria-label="Document">
      <div className="doc-tools">
        <Segmented<Overlay> label="Overlay" value={overlay} onChange={(value) => { setOverlay(value); setSplit(null); }} options={overlayOptions} />
        {original && (
          <button type="button" className={`btn sm ${split !== null ? "primary" : ""}`} onClick={() => setSplit(split === null ? 50 : null)} title="Compare with the image before cleanup">
            <SplitSquareHorizontal size={14} />Before / after
          </button>
        )}
        <span className="zoom">
          <button type="button" className="icon-btn sm" aria-label="Zoom out" disabled={zoom <= 0.5} onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.25).toFixed(2)))}><ZoomOut size={15} /></button>
          <button type="button" className="btn sm ghost zoom-val" onClick={() => setZoom(1)} title="Fit to width">{zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}</button>
          <button type="button" className="icon-btn sm" aria-label="Zoom in" disabled={zoom >= 4} onClick={() => setZoom((z) => Math.min(4, +(z + 0.25).toFixed(2)))}><ZoomIn size={15} /></button>
        </span>
      </div>

      <div className="stage" ref={stageRef} onMouseLeave={() => setTip(null)}>
        <div className="canvas" style={{ width: `${zoom * 100}%`, maxWidth: zoom === 1 ? Math.max(360, Math.min(900, (page.width / page.height) * 780)) : undefined }}>
          {split !== null && original ? (
            <div className="paper compare-wrap">
              <img src={`data:image/jpeg;base64,${original.image}`} alt={`Page ${page.page_number} before cleanup`} />
              <div className="after" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
                <img src={`data:image/jpeg;base64,${page.preview_image}`} alt={`Page ${page.page_number} after cleanup`} />
              </div>
              <span className="split-line" style={{ left: `${split}%` }}><i /></span>
              <input className="split-range" type="range" min={0} max={100} value={split} aria-label="Before and after position" onChange={(event) => setSplit(Number(event.target.value))} />
              <span className="split-tag" style={{ left: 10 }}>Before</span>
              <span className="split-tag" style={{ right: 10 }}>After cleanup</span>
            </div>
          ) : (
            <div className="paper">
              <img src={`data:image/jpeg;base64,${page.preview_image}`} alt={`Document page ${page.page_number}`} />
              {overlay !== "off" && (
                <svg className="ov" viewBox={`0 0 ${page.width} ${page.height}`} preserveAspectRatio="none" role="group" aria-label="Recognised regions">
                  {overlay === "confidence" && page.regions.map((region, index) => (
                    <polygon
                      key={`r${index}`}
                      points={points(region.polygon)}
                      fill={levelColor(region.confidence, 0.14)}
                      stroke={levelColor(region.confidence, 0.9)}
                      strokeWidth={2.5}
                      onMouseMove={(event) => showTip(event, "Recognised region", region.text, region.confidence, "confidence")}
                    />
                  ))}
                  {overlay !== "confidence" && blocks.map((block, index) => {
                    const agree = agreement(block);
                    const color = overlay === "agreement"
                      ? (agree === null ? "#98a2b3" : levelColor(agree, 0.9))
                      : KIND_COLORS[block.kind] ?? "#2e90fa";
                    const fill = overlay === "agreement"
                      ? (agree === null ? "rgba(152,162,179,.1)" : levelColor(agree, 0.14))
                      : `${KIND_COLORS[block.kind] ?? "#2e90fa"}1a`;
                    const label = KIND_LABELS[block.kind] ?? block.kind;
                    return (
                      <polygon
                        key={`b${index}`}
                        points={points(block.polygon)}
                        fill={fill}
                        stroke={color}
                        strokeWidth={2.5}
                        strokeDasharray={overlay === "layout" && block.review_required ? "8 5" : undefined}
                        role="button"
                        tabIndex={0}
                        aria-label={`${label} ${index + 1}: ${block.text}`}
                        onClick={() => onSelect(selected === index ? null : index)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            onSelect(index);
                          }
                        }}
                        onMouseMove={(event) => showTip(
                          event,
                          label,
                          block.kind === "illegible" ? block.raw_text ?? "" : block.text,
                          overlay === "agreement" ? agree : block.confidence,
                          overlay === "agreement" ? "of words agreed" : "confidence",
                        )}
                      />
                    );
                  })}
                  {selected !== null && blocks[selected] && overlay !== "confidence" && (
                    <polygon points={points(blocks[selected].polygon)} fill="rgba(79,70,229,.10)" stroke="#4f46e5" strokeWidth={5} pointerEvents="none" />
                  )}
                </svg>
              )}
            </div>
          )}
        </div>
        {tip && (
          <div className="tip" style={{ left: tip.x, top: tip.y }}>
            <b>{tip.title}</b>{tip.value !== null && <> · {Math.round(tip.value * 100)}% {tip.valueLabel}</>}
            <span className="tip-sub">{tip.text || "—"}</span>
          </div>
        )}
      </div>

      <div className="doc-foot">
        <span className="legend">
          {overlay === "agreement" && <><i className="lg" style={{ background: "#17b26a" }} />Models agree<i className="lg" style={{ background: "#f79009" }} />Partly<i className="lg" style={{ background: "#f04438" }} />Disputed</>}
          {overlay === "confidence" && <><i className="lg" style={{ background: "#17b26a" }} />High<i className="lg" style={{ background: "#f79009" }} />Medium<i className="lg" style={{ background: "#f04438" }} />Low confidence</>}
          {overlay === "layout" && Object.entries(KIND_LABELS)
            .filter(([kind]) => kind !== "title" && blocks.some((b) => b.kind === kind || (kind === "heading" && b.kind === "title")))
            .map(([kind, text]) => <span key={kind}><i className="lg" style={{ background: KIND_COLORS[kind] }} />{text}</span>)}
          {overlay === "off" && <span className="muted">Overlays hidden</span>}
        </span>
        <span className="muted">Hover a region to read it. Click to jump to the line.</span>
      </div>
    </section>
  );
}
