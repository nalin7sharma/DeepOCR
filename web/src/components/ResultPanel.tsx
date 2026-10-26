import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, BookOpen, Braces, Check, Copy, Download, FileCode2, ListTree, PenLine, RotateCcw } from "lucide-react";
import type { OcrResponse, StructuredBlock } from "../types";
import { baseName, confidenceColor, plainText, saveFile } from "../lib/text";
import { KIND_COLORS, KIND_LABELS } from "./DocumentViewer";
import { MarkdownView } from "./MarkdownView";

type Tab = "reader" | "editor" | "lines" | "review" | "json";

interface Props {
  result: OcrResponse;
  draft: string;
  onDraft: (value: string) => void;
  pageIndex: number;
  selected: number | null;
  onSelect: (index: number | null, block?: StructuredBlock) => void;
  onError: (message: string) => void;
}

export function extractPageMarkdown(markdown: string, pageNumber: number): string | null {
  const heading = new RegExp(
    `(?:^|\\n)##\\s+Page\\s+${pageNumber}\\s*\\n([\\s\\S]*?)(?=\\n##\\s+Page\\s+\\d+\\s*(?:\\n|$)|$)`,
  );
  const match = markdown.match(heading);
  return match ? match[1].trim() : null;
}

export function ResultPanel({ result, draft, onDraft, pageIndex, selected, onSelect, onError }: Props) {
  const [tab, setTab] = useState<Tab>("reader");
  const [copied, setCopied] = useState(false);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const lineRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const page = result.pages[pageIndex];
  const reviewItems = useMemo(
    () => result.pages.flatMap((p, pi) => p.blocks.map((block, bi) => ({ block, pi, bi })).filter(({ block }) => block.review_required)),
    [result],
  );
  const edited = draft !== result.markdown;

  // Selecting a region on the page scrolls / highlights the matching line or editor text.
  useEffect(() => {
    if (selected === null || !page) return;
    const block = page.blocks[selected];
    if (tab === "lines") lineRefs.current[selected]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (tab === "editor" && block) {
      const start = draft.indexOf(block.markdown.trim());
      if (start >= 0) {
        editorRef.current?.focus();
        editorRef.current?.setSelectionRange(start, start + block.markdown.trim().length);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, pageIndex, tab]);

  async function copyText() {
    try {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      onError("Clipboard access was blocked. Select and copy the text from the editor.");
    }
  }

  function exportJson() {
    const output = {
      ...result,
      text: draft,
      markdown: draft,
      edited,
      pages: result.pages.map(({ preview_image: _preview, preprocessing, ...item }) => {
        const pageText = extractPageMarkdown(draft, item.page_number) ?? item.markdown;
        return {
          ...item,
          preprocessing: preprocessing ? { applied: preprocessing.applied, deskew_angle: preprocessing.deskew_angle } : undefined,
          text: pageText,
          markdown: pageText,
        };
      }),
    };
    saveFile(`${baseName(result.filename)}.json`, JSON.stringify(output, null, 2), "application/json");
  }

  const jsonPreview = useMemo(() => JSON.stringify(
    { ...result, pages: result.pages.map(({ preview_image: _p, preprocessing: _s, ...rest }) => rest) },
    null,
    2,
  ).slice(0, 60_000), [result]);

  const TABS: { id: Tab; label: string; icon: typeof BookOpen; count?: number }[] = [
    { id: "reader", label: "Reader", icon: BookOpen },
    { id: "editor", label: "Editor", icon: PenLine },
    { id: "lines", label: "Lines", icon: ListTree, count: page?.blocks.length },
    { id: "review", label: "Review", icon: AlertTriangle, count: reviewItems.length },
    { id: "json", label: "JSON", icon: Braces },
  ];

  return (
    <div className="result-panel">
      <div className="result-tabs" role="tablist" aria-label="Extraction views">
        {TABS.map(({ id, label, icon: Icon, count }) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id} className={`result-tab ${tab === id ? "on" : ""}`} onClick={() => setTab(id)}>
            {tab === id && <motion.span layoutId="result-tab" className="result-tab-pill" />}
            <Icon size={14} /> {label}
            {count ? <b className={id === "review" ? "warn" : ""}>{count}</b> : null}
          </button>
        ))}
        {edited && <span className="edited-flag"><PenLine size={12} /> edited</span>}
      </div>

      <div className="result-body">
        <AnimatePresence mode="wait">
          <motion.div key={tab} className="result-view" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.16 }}>
            {tab === "reader" && (draft.trim() ? <MarkdownView markdown={draft} /> : <p className="empty-note">No text was recognised on this document.</p>)}

            {tab === "editor" && (
              <div className="editor-wrap">
                <textarea
                  id="markdown-editor"
                  ref={editorRef}
                  className="editor"
                  aria-label="Editable extracted Markdown"
                  value={draft}
                  spellCheck
                  onChange={(event) => onDraft(event.target.value)}
                />
                <div className="editor-foot">
                  <span>{draft.length.toLocaleString()} chars · {draft.split(/\s+/).filter(Boolean).length.toLocaleString()} words · Markdown</span>
                  <button type="button" className="ghost-button small" disabled={!edited} onClick={() => onDraft(result.markdown)}><RotateCcw size={13} /> Reset to OCR</button>
                </div>
              </div>
            )}

            {tab === "lines" && page && (
              <div className="line-list">
                {page.blocks.length === 0 && <p className="empty-note">No lines on this page.</p>}
                {page.blocks.map((block, index) => (
                  <button
                    key={index}
                    ref={(element) => { lineRefs.current[index] = element; }}
                    type="button"
                    className={`line-item ${selected === index ? "on" : ""} ${block.review_required ? "review" : ""}`}
                    onClick={() => onSelect(selected === index ? null : index, block)}
                  >
                    <span className="line-kind" style={{ ["--kind" as string]: KIND_COLORS[block.kind] ?? "#22d3ee" }}>{KIND_LABELS[block.kind] ?? block.kind}</span>
                    <span className="line-text">
                      {block.kind === "table" && block.cells ? (
                        <span className="mini-table">{block.cells.slice(0, 4).map((row, r) => <span key={r}>{row.join("  ·  ")}</span>)}</span>
                      ) : block.original_text ? (
                        <><del>{block.original_text}</del><ins>{block.text}</ins></>
                      ) : (
                        block.raw_text || block.text
                      )}
                    </span>
                    <span className="line-conf" title={`Confidence ${Math.round(block.confidence * 100)}%`}>
                      <i style={{ width: `${block.confidence * 100}%`, background: confidenceColor(block.confidence) }} />
                      <em>{Math.round(block.confidence * 100)}</em>
                    </span>
                  </button>
                ))}
              </div>
            )}

            {tab === "review" && (
              <div className="line-list">
                {reviewItems.length === 0 ? (
                  <div className="all-clear"><Check size={22} /><strong>Nothing flagged</strong><span>No uncertain text, tables, checkboxes or signature fields need a closer look.</span></div>
                ) : reviewItems.map(({ block, pi, bi }) => (
                  <button key={`${pi}-${bi}`} type="button" className="line-item review" onClick={() => onSelect(bi, block)}>
                    <span className="line-kind" style={{ ["--kind" as string]: KIND_COLORS[block.kind] ?? "#f87171" }}>{KIND_LABELS[block.kind] ?? block.kind}</span>
                    <span className="line-text">
                      {block.kind === "illegible" ? <>Low confidence: <q>{block.raw_text}</q></> : block.kind === "table" ? "Inferred table — check row/column layout" : block.kind === "signature" ? `Labelled signature field: ${block.text}` : block.checkbox_state === "uncertain" ? `Unclear checkbox: ${block.text}` : block.text}
                      {result.pages.length > 1 && <small> · page {pi + 1}</small>}
                    </span>
                    <span className="line-conf"><i style={{ width: `${block.confidence * 100}%`, background: confidenceColor(block.confidence) }} /><em>{Math.round(block.confidence * 100)}</em></span>
                  </button>
                ))}
              </div>
            )}

            {tab === "json" && <pre className="json-view">{jsonPreview}</pre>}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="export-bar">
        <button type="button" className="ghost-button" onClick={copyText}>{copied ? <><Check size={15} /> Copied</> : <><Copy size={15} /> Copy</>}</button>
        <div className="export-actions">
          <button type="button" className="ghost-button" onClick={() => saveFile(`${baseName(result.filename)}.txt`, plainText(draft), "text/plain;charset=utf-8")}>.TXT</button>
          <button type="button" className="ghost-button" onClick={exportJson}><FileCode2 size={15} /> JSON</button>
          <button type="button" className="primary-button" onClick={() => saveFile(`${baseName(result.filename)}.md`, draft, "text/markdown;charset=utf-8")}>
            <Download size={15} /> Export Markdown
          </button>
        </div>
      </div>
    </div>
  );
}
