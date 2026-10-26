import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Braces, Check, CheckCircle2, Copy, Download, ListOrdered, LocateFixed, Pencil, RotateCcw, Text, Wand2, X } from "lucide-react";
import type { OcrResponse, StructuredBlock, WordVote } from "../types";
import {
  agreement, docMarkdown, editableText, isDisputed, reviewItems, tokensWithVotes, type DocState, type ReviewItem,
} from "../lib/document";
import { baseName, level, plainText, saveFile } from "../lib/text";
import { shortName } from "../lib/engines";
import { KIND_COLORS, KIND_LABELS } from "./DocumentViewer";
import { MarkdownView } from "./MarkdownView";
import { Segmented, useToast } from "./Ui";

export type Tab = "text" | "lines" | "review" | "json";

export interface DocActions {
  setText: (pageIndex: number, blockIndex: number, text: string) => void;
  replaceWord: (pageIndex: number, blockIndex: number, tokenIndex: number, word: string) => void;
  resolve: (key: string) => void;
  setMarkdown: (markdown: string | null) => void;
}

interface Props {
  result: OcrResponse;
  doc: DocState;
  pageIndex: number;
  tab: Tab;
  onTab: (tab: Tab) => void;
  selected: number | null;
  onSelect: (pageIndex: number, blockIndex: number | null) => void;
  actions: DocActions;
}

/* ------------------------------------------------------------ exports */
export function useExports(result: OcrResponse, doc: DocState) {
  const toast = useToast();
  const markdown = docMarkdown(doc);
  const name = baseName(result.filename);
  return {
    copy: async () => {
      try {
        await navigator.clipboard.writeText(markdown);
        toast("Copied to the clipboard");
      } catch {
        toast("Clipboard blocked: select the text in the Markdown view instead");
      }
    },
    markdown: () => {
      saveFile(`${name}.md`, markdown, "text/markdown;charset=utf-8");
      toast(`Saved ${name}.md`);
    },
    text: () => {
      saveFile(`${name}.txt`, plainText(markdown), "text/plain;charset=utf-8");
      toast(`Saved ${name}.txt`);
    },
    json: () => {
      saveFile(`${name}.json`, JSON.stringify(exportJson(result, doc), null, 2), "application/json");
      toast(`Saved ${name}.json`);
    },
  };
}

export function exportJson(result: OcrResponse, doc: DocState) {
  return {
    ...result,
    markdown: docMarkdown(doc),
    text: docMarkdown(doc),
    reviewed: true,
    pages: result.pages.map(({ preview_image: _preview, preprocessing, ...page }, index) => {
      const blocks = doc.pages[index].blocks;
      const markdown = blocks.map((block) => block.markdown).join("\n");
      return {
        ...page,
        blocks,
        markdown,
        text: markdown,
        preprocessing: preprocessing ? { applied: preprocessing.applied, deskew_angle: preprocessing.deskew_angle } : undefined,
      };
    }),
  };
}

/* ------------------------------------------------------------ disputed word */
function WordPopover({ vote, word, onUse, onKeep, onClose }: {
  vote: WordVote;
  word: string;
  onUse: (word: string) => void;
  onKeep: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [shift, setShift] = useState(0);
  // Keep the popover inside the text panel when the word sits near its right edge.
  useLayoutEffect(() => {
    const element = ref.current;
    const bounds = element?.closest(".tbody")?.getBoundingClientRect();
    if (!element || !bounds) return;
    const box = element.getBoundingClientRect();
    const overflow = box.right - (bounds.right - 8);
    if (overflow > 0) setShift(-Math.min(overflow, box.left - bounds.left - 8));
  }, []);
  useEffect(() => {
    const onDown = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) onClose();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);
  const alternatives = Object.entries(vote.alternatives).sort((a, b) => b[1] - a[1]).slice(0, 4);
  return (
    <span className="pop" ref={ref} role="dialog" aria-label={`Alternatives for ${word}`} style={shift ? { transform: `translateX(${shift}px)` } : undefined} onClick={(event) => event.stopPropagation()}>
      <span className="pop-h">The models disagreed here</span>
      <span className="pop-row chosen"><b>{word}</b><span className="num">{Math.round((vote.support ?? 0) * 100)}%</span><button type="button" className="btn sm ghost" onClick={onKeep}>Keep</button></span>
      {alternatives.map(([alternative, share]) => (
        <button key={alternative} type="button" className="pop-row" onClick={() => onUse(alternative)}>
          <b>{alternative}</b><span className="num">{Math.round(share * 100)}%</span><em>Use this</em>
        </button>
      ))}
      <span className="pop-src">Picked by {vote.sources.map((s) => shortName(s)).join(", ") || "—"}. Vote share = model weight × its confidence.</span>
    </span>
  );
}

function LineText({ block, pageIndex, blockIndex, openKey, setOpenKey, actions }: {
  block: StructuredBlock;
  pageIndex: number;
  blockIndex: number;
  openKey: string | null;
  setOpenKey: (key: string | null) => void;
  actions: DocActions;
}) {
  if (block.kind === "table" && block.cells) {
    return <span className="code" style={{ fontSize: 12 }}>{block.cells.slice(0, 4).map((row) => row.join(" · ")).join(" / ")}{block.cells.length > 4 ? " …" : ""}</span>;
  }
  if (block.kind === "illegible") return <span className="muted">{block.raw_text || "—"}</span>;
  const tokens = tokensWithVotes(block);
  return (
    <>
      {tokens.map(({ text, vote }, t) => {
        const key = `${pageIndex}:${blockIndex}:${t}`;
        const fixed = vote?.sources.includes("reviewer");
        const node = vote && isDisputed(vote) ? (
          <span
            key={key}
            className={`dis ${openKey === key ? "open" : ""}`}
            role="button"
            tabIndex={0}
            aria-label={`${text}: models disagreed, show alternatives`}
            onClick={(event) => {
              event.stopPropagation();
              setOpenKey(openKey === key ? null : key);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setOpenKey(key);
              }
            }}
          >
            {text}
            {openKey === key && (
              <WordPopover
                vote={vote}
                word={text}
                onUse={(word) => {
                  actions.replaceWord(pageIndex, blockIndex, t, word);
                  setOpenKey(null);
                }}
                onKeep={() => {
                  actions.resolve(key);
                  actions.replaceWord(pageIndex, blockIndex, t, text);
                  setOpenKey(null);
                }}
                onClose={() => setOpenKey(null)}
              />
            )}
          </span>
        ) : <span key={key} className={fixed ? "fixed-word" : undefined}>{text}</span>;
        return [t > 0 ? " " : null, node];
      })}
    </>
  );
}

/* ------------------------------------------------------------ lines */
function LinesView({ doc, pageIndex, selected, onSelect, actions }: {
  doc: DocState;
  pageIndex: number;
  selected: number | null;
  onSelect: (pageIndex: number, blockIndex: number | null) => void;
  actions: DocActions;
}) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const refs = useRef<(HTMLDivElement | null)[]>([]);
  const blocks = doc.pages[pageIndex].blocks;
  const consensus = blocks.some((b) => b.words?.some((w) => w.support !== null));

  useEffect(() => {
    if (selected !== null) refs.current[selected]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  if (!blocks.length) return <p className="empty">No text was found on this page.</p>;
  return (
    <>
      <p className="tb-head">
        {consensus
          ? <><span className="dis" style={{ cursor: "default" }}>Underlined</span> words had no majority among the models. Click one to see the alternatives.</>
          : "Click a line to find it on the page. Use the pencil to correct it."}
      </p>
      {blocks.map((block, index) => {
        const agree = agreement(block);
        const score = agree ?? block.confidence;
        const kindColor = KIND_COLORS[block.kind];
        return (
          <div
            key={index}
            ref={(el) => { refs.current[index] = el; }}
            className={`ln ${selected === index ? "sel" : ""}`}
            onClick={() => editing === null && onSelect(pageIndex, selected === index ? null : index)}
          >
            <span className="ln-n">{index + 1}</span>
            <span className="ln-t">
              {block.kind !== "text" && <span className="ln-kind" style={{ ["--kc" as string]: kindColor }}>{KIND_LABELS[block.kind] ?? block.kind}</span>}
              {editing === index ? (
                <span onClick={(event) => event.stopPropagation()} style={{ display: "block" }}>
                  <textarea
                    className="ln-edit"
                    aria-label={`Edit line ${index + 1}`}
                    value={draft}
                    autoFocus
                    onChange={(event) => setDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                        actions.setText(pageIndex, index, draft);
                        setEditing(null);
                      }
                      if (event.key === "Escape") setEditing(null);
                    }}
                  />
                  <span className="ln-edit-actions">
                    <button type="button" className="btn sm primary" onClick={() => { actions.setText(pageIndex, index, draft); setEditing(null); }}><Check size={13} />Save</button>
                    <button type="button" className="btn sm" onClick={() => setEditing(null)}><X size={13} />Cancel</button>
                    <span className="hint" style={{ alignSelf: "center" }}>Ctrl + Enter to save</span>
                  </span>
                </span>
              ) : (
                <LineText block={block} pageIndex={pageIndex} blockIndex={index} openKey={openKey} setOpenKey={setOpenKey} actions={actions} />
              )}
              {(block.corrections?.length ?? 0) > 0 && editing !== index && (
                <span className="fixmark" title={block.corrections!.map((c) => `${c.original} → ${c.corrected}`).join(", ")}><Wand2 size={11} />auto-fixed</span>
              )}
            </span>
            <span className="ln-side">
              {editing !== index && (
                <button
                  type="button"
                  className="icon-btn sm"
                  aria-label={`Edit line ${index + 1}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    setDraft(editableText(block));
                    setEditing(index);
                  }}
                ><Pencil size={13} /></button>
              )}
              <span className={`conf ${level(score)}`} title={agree !== null ? "Share of words the models agreed on" : "Recognition confidence"}>{Math.round(score * 100)}%</span>
            </span>
          </div>
        );
      })}
    </>
  );
}

/* ------------------------------------------------------------ review */
function ReviewView({ doc, actions, onSelect }: { doc: DocState; actions: DocActions; onSelect: (pageIndex: number, blockIndex: number | null) => void }) {
  const items = reviewItems(doc);
  if (!items.length) {
    return (
      <div className="done-state">
        <CheckCircle2 size={28} />
        <b>Everything is reviewed</b>
        <span>No disputed words, unclear lines, tables, checkboxes or signature fields are waiting.</span>
      </div>
    );
  }
  const actionsFor = (item: ReviewItem) => {
    const block = doc.pages[item.pageIndex].blocks[item.blockIndex];
    if (item.kind === "word" && item.vote && item.tokenIndex !== undefined) {
      const alternatives = Object.entries(item.vote.alternatives).sort((a, b) => b[1] - a[1]).slice(0, 3);
      return (
        <>
          <button type="button" className="btn sm" onClick={() => { actions.resolve(item.key); actions.replaceWord(item.pageIndex, item.blockIndex, item.tokenIndex!, item.detail); }}><Check size={13} />Keep “{item.detail}”</button>
          {alternatives.map(([word, share]) => (
            <button key={word} type="button" className="btn sm" onClick={() => actions.replaceWord(item.pageIndex, item.blockIndex, item.tokenIndex!, word)}>
              Use “{word}” <span className="muted num">{Math.round(share * 100)}%</span>
            </button>
          ))}
        </>
      );
    }
    if (item.kind === "illegible") {
      return (
        <button type="button" className="btn sm" onClick={() => actions.setText(item.pageIndex, item.blockIndex, block.raw_text ?? "")}><Check size={13} />Use the OCR text</button>
      );
    }
    return <button type="button" className="btn sm" onClick={() => actions.resolve(item.key)}><Check size={13} />Looks right</button>;
  };
  return (
    <div className="rq">
      <p className="tb-head">{items.length} {items.length === 1 ? "item needs" : "items need"} a human check. Compare each with the page, then confirm or fix it.</p>
      {items.map((item) => {
        const block = doc.pages[item.pageIndex].blocks[item.blockIndex];
        return (
          <div key={item.key} className="rq-item">
            <div className="rq-top">
              <AlertTriangle size={14} color="#f79009" />
              <b>{item.title}</b>
              <span>· line {item.blockIndex + 1}{doc.pages.length > 1 ? `, page ${doc.pages[item.pageIndex].page_number}` : ""}</span>
              <button type="button" className="btn sm ghost" style={{ marginLeft: "auto" }} onClick={() => onSelect(item.pageIndex, item.blockIndex)}><LocateFixed size={13} />Show</button>
            </div>
            <div className="rq-text">
              {item.kind === "word"
                ? <>…{tokensWithVotes(block).map(({ text }, t) => (t === item.tokenIndex ? <mark key={t} className="dis" style={{ background: "#fffaeb" }}>{text}</mark> : <span key={t}>{text}</span>)).reduce<React.ReactNode[]>((acc, node, i) => (i ? [...acc, " ", node] : [node]), [])}…</>
                : item.detail || "—"}
            </div>
            <div className="rq-actions">{actionsFor(item)}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------ panel */
export function TextPanel({ result, doc, pageIndex, tab, onTab, selected, onSelect, actions }: Props) {
  const [view, setView] = useState<"preview" | "markdown">("preview");
  const exports = useExports(result, doc);
  const markdown = docMarkdown(doc);
  const items = reviewItems(doc);
  const json = useMemo(() => JSON.stringify(exportJson(result, doc), null, 2).slice(0, 80_000), [result, doc]);
  const lines = doc.pages[pageIndex].blocks.length;

  const tabs: { id: Tab; label: string; icon: typeof Text; count?: number; warn?: boolean }[] = [
    { id: "text", label: "Text", icon: Text },
    { id: "lines", label: "Lines", icon: ListOrdered, count: lines },
    { id: "review", label: "Review", icon: AlertTriangle, count: items.length, warn: items.length > 0 },
    { id: "json", label: "JSON", icon: Braces },
  ];

  return (
    <section className="card textp" aria-label="Extracted text">
      <div className="ttabs" role="tablist" aria-label="Text views">
        {tabs.map(({ id, label, icon: Icon, count, warn }) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id} className={`ttab ${tab === id ? "on" : ""}`} onClick={() => onTab(id)}>
            <Icon size={14} />{label}{count !== undefined && <span className={`badge ${warn ? "warning" : ""}`}>{count}</span>}
          </button>
        ))}
      </div>
      {tab === "text" && (
        <div className="subbar">
          <Segmented<"preview" | "markdown"> small label="Text view" value={view} onChange={setView} options={[{ value: "preview", label: "Preview" }, { value: "markdown", label: "Markdown" }]} />
          {doc.markdownOverride !== null
            ? <button type="button" className="btn sm ghost" onClick={() => actions.setMarkdown(null)}><RotateCcw size={13} />Discard Markdown edits</button>
            : <span className="muted">Edits here update every export</span>}
        </div>
      )}
      <div className="tbody">
        {tab === "text" && view === "preview" && (markdown.trim() ? <MarkdownView markdown={markdown} /> : <p className="empty">No text was recognised.</p>)}
        {tab === "text" && view === "markdown" && (
          <>
            {doc.markdownOverride !== null && <div className="alert info" style={{ marginBottom: 10 }}>You are editing the Markdown directly, so line edits and word swaps are paused until you discard these edits.</div>}
            <textarea className="editor" aria-label="Editable Markdown" value={markdown} spellCheck onChange={(event) => actions.setMarkdown(event.target.value)} />
          </>
        )}
        {tab === "lines" && <LinesView doc={doc} pageIndex={pageIndex} selected={selected} onSelect={onSelect} actions={actions} />}
        {tab === "review" && <ReviewView doc={doc} actions={actions} onSelect={(p, b) => { onSelect(p, b); onTab("lines"); }} />}
        {tab === "json" && <pre className="json-view">{json}</pre>}
      </div>
      <div className="tfoot">
        <button type="button" className="btn sm" onClick={exports.copy}><Copy size={14} />Copy</button>
        <button type="button" className="btn sm" onClick={exports.text}>.txt</button>
        <button type="button" className="btn sm" onClick={exports.json}>JSON</button>
        <button type="button" className="btn sm primary" onClick={exports.markdown}><Download size={14} />Markdown</button>
      </div>
    </section>
  );
}
