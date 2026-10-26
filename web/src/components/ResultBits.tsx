import { useState } from "react";
import { AlertTriangle, Undo2, Wand2 } from "lucide-react";
import type { OcrResponse } from "../types";
import type { DocState } from "../lib/document";
import { allCorrections, isDisputed, reviewItems } from "../lib/document";

const STAGE_COLORS: [string, string][] = [
  ["decode", "#98a2b3"], ["preprocess", "#ee46bc"], ["recognition", "#4f46e5"], ["layout", "#06aed4"], ["postprocess", "#16b364"],
];

/** Compact metrics strip; consensus results show agreement instead of confidence. */
export function SummaryStrip({ result, doc, models = 5 }: { result: OcrResponse; doc: DocState; models?: number }) {
  const words = doc.pages.flatMap((page) => page.blocks.flatMap((block) => block.words ?? [])).filter((w) => w.support !== null);
  const consensus = words.length > 0;
  const total = result.timings?.total ?? result.inference_ms;
  const items = reviewItems(doc);
  const disputed = words.filter((w) => isDisputed(w)).length;
  const agree = consensus ? (words.length - disputed) / words.length : result.stats?.mean_confidence ?? 0;
  const fixes = allCorrections(doc).length;
  const cells: { label: string; value: string | number; warn?: boolean; extra?: JSX.Element }[] = [
    {
      label: consensus ? "Agreement" : "Confidence",
      value: `${Math.round(agree * 100)}%`,
      extra: <span className="mini"><i style={{ width: `${agree * 100}%` }} /></span>,
    },
    { label: "Lines", value: doc.pages.reduce((sum, page) => sum + page.blocks.length, 0) },
    { label: "Words", value: result.stats?.words ?? 0 },
    consensus ? { label: "Disputed words", value: disputed, warn: disputed > 0 } : { label: "To review", value: items.length, warn: items.length > 0 },
    consensus ? { label: "Models voting", value: models } : { label: "Auto-fixes", value: fixes },
    {
      label: "Time",
      value: total >= 1000 ? `${(total / 1000).toFixed(1)} s` : `${Math.round(total)} ms`,
      extra: (
        <span className="stack" title="Time per stage">
          {STAGE_COLORS.map(([key, color]) => {
            const value = result.timings?.[key] ?? 0;
            return value > 0 ? <i key={key} style={{ width: `${(value / total) * 100}%`, background: color }} title={`${key}: ${Math.round(value)} ms`} /> : null;
          })}
        </span>
      ),
    },
  ];
  return (
    <section className="card strip" aria-label="Extraction summary">
      {cells.map((cell) => (
        <div className="sv" key={cell.label}>
          <span className="label">{cell.label}</span>
          <b className={`num ${cell.warn ? "warn" : ""}`}>{cell.value}</b>
          {cell.extra}
        </div>
      ))}
    </section>
  );
}

/** Automatic fixes (each undoable) and the review count, above the workspace. */
export function FixBar({ doc, onUndo, onReview }: {
  doc: DocState;
  onUndo: (pageIndex: number, blockIndex: number, index: number) => void;
  onReview: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const fixes = allCorrections(doc);
  const items = reviewItems(doc);
  if (!fixes.length && !items.length) return null;
  const shown = showAll ? fixes : fixes.slice(0, 1);
  const kinds = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.kind] = (acc[item.kind] ?? 0) + 1;
    return acc;
  }, {});
  const KIND: Record<string, [string, string]> = {
    word: ["disputed word", "disputed words"],
    checkbox: ["checkbox", "checkboxes"],
    table: ["table", "tables"],
    signature: ["signature field", "signature fields"],
    illegible: ["unclear line", "unclear lines"],
    other: ["line", "lines"],
  };
  const summary = Object.entries(kinds).map(([kind, n]) => `${n} ${KIND[kind][n === 1 ? 0 : 1]}`).join(", ");
  return (
    <div className="fixbar" role="region" aria-label="Automatic fixes and review">
      {fixes.length > 0 && (
        <span className="fx">
          <Wand2 size={15} />
          <span><b>{fixes.length} automatic {fixes.length === 1 ? "fix" : "fixes"}:</b></span>
          {shown.map(({ pageIndex, blockIndex, correction }) => {
            const block = doc.pages[pageIndex].blocks[blockIndex];
            const index = (block.corrections ?? []).indexOf(correction);
            return (
              <span key={`${pageIndex}-${blockIndex}-${correction.position}-${correction.corrected}`} className="fx">
                <del>{correction.original}</del>→<ins>{correction.corrected}</ins>
                <button type="button" className="btn sm ghost" onClick={() => onUndo(pageIndex, blockIndex, index)}><Undo2 size={13} />Undo</button>
              </span>
            );
          })}
          {fixes.length > 1 && (
            <button type="button" className="btn sm ghost" onClick={() => setShowAll(!showAll)}>{showAll ? "Show less" : `+${fixes.length - 1} more`}</button>
          )}
        </span>
      )}
      {fixes.length > 0 && items.length > 0 && <span className="sep" />}
      {items.length > 0 && (
        <span className="fx rv">
          <AlertTriangle size={15} />
          <span><b>{items.length} to review:</b> {summary}</span>
          <button type="button" className="btn sm" onClick={onReview}>Start review</button>
        </span>
      )}
    </div>
  );
}
