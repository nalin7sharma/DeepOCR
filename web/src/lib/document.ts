/**
 * Editable document model for the results workspace.
 *
 * The OCR result's blocks are copied into state the reviewer can change: edit a
 * line, swap a disputed word for an alternative another model read, undo an
 * automatic fix, or mark a flagged item as checked. Markdown for export is
 * always derived from these blocks, unless the reviewer edits the raw Markdown
 * directly, which then takes precedence until it is reset.
 */
import type { Correction, OcrResponse, StructuredBlock, WordVote } from "../types";

export interface DocState {
  pages: { page_number: number; blocks: StructuredBlock[] }[];
  /** Raw Markdown typed by the reviewer; overrides the block-derived Markdown. */
  markdownOverride: string | null;
  /** Review items the reviewer has marked as checked ("p:b" for blocks, "p:b:w" for words). */
  resolved: string[];
}

/** A word is disputed when no reading won a majority of the vote. */
export const DISPUTE_THRESHOLD = 0.5;

export const isDisputed = (word: WordVote | undefined | null) =>
  !!word && word.support !== null && word.support < DISPUTE_THRESHOLD;

export function initDoc(result: OcrResponse): DocState {
  return {
    pages: result.pages.map((page) => ({
      page_number: page.page_number,
      blocks: page.blocks.map((block) => ({ ...block, corrections: [...(block.corrections ?? [])], words: block.words ? block.words.map((w) => ({ ...w })) : block.words })),
    })),
    markdownOverride: null,
    resolved: [],
  };
}

export function pageMarkdown(state: DocState, pageIndex: number): string {
  return state.pages[pageIndex].blocks.map((block) => block.markdown).join("\n");
}

export function docMarkdown(state: DocState): string {
  if (state.markdownOverride !== null) return state.markdownOverride;
  return state.pages
    .map((page, index) => `## Page ${page.page_number}\n\n${pageMarkdown(state, index)}`.trimEnd())
    .join("\n\n");
}

/** Markdown for a block whose visible text changed, keeping its structural prefix/suffix. */
function markdownFor(block: StructuredBlock, text: string): string {
  if (block.kind === "heading" || block.kind === "title") {
    const hashes = /^(#{1,6})\s/.exec(block.markdown)?.[1] ?? (block.kind === "title" ? "#" : "##");
    return `${hashes} ${text}`;
  }
  if (block.kind === "signature" && block.markdown.endsWith(" [Signature]")) return `${text} [Signature]`;
  return text;
}

function updateBlock(state: DocState, pageIndex: number, blockIndex: number, change: (block: StructuredBlock) => StructuredBlock): DocState {
  return {
    ...state,
    pages: state.pages.map((page, p) => (p !== pageIndex ? page : {
      ...page,
      blocks: page.blocks.map((block, b) => (b === blockIndex ? change(block) : block)),
    })),
  };
}

/** The text the reviewer sees and edits for a block. */
export function editableText(block: StructuredBlock): string {
  if (block.kind === "illegible") return block.raw_text ?? "";
  if (block.kind === "table") return block.markdown;
  return block.text;
}

/** Replace a line's text. An edited line counts as reviewed, so its votes are dropped. */
export function setBlockText(state: DocState, pageIndex: number, blockIndex: number, text: string): DocState {
  const clean = text.replace(/\s+\n/g, "\n").trim();
  return updateBlock(state, pageIndex, blockIndex, (block) => {
    if (block.kind === "illegible") {
      return { ...block, kind: "text", text: clean, markdown: clean, review_required: false, words: null };
    }
    if (block.kind === "table") return { ...block, text: clean, markdown: clean, review_required: false };
    return { ...block, text: clean, markdown: markdownFor(block, clean), words: null };
  });
}

/** Tokens of a block's text paired with their votes when they line up one-to-one. */
export function tokensWithVotes(block: StructuredBlock): { text: string; vote: WordVote | null }[] {
  const tokens = block.text.split(/\s+/).filter(Boolean);
  const votes = block.words ?? null;
  const aligned = !!votes && votes.length === tokens.length;
  return tokens.map((text, i) => ({ text, vote: aligned ? (votes as WordVote[])[i] : null }));
}

/** Swap one word for an alternative (or a typed word); the word is then settled. */
export function replaceWord(state: DocState, pageIndex: number, blockIndex: number, tokenIndex: number, word: string): DocState {
  return updateBlock(state, pageIndex, blockIndex, (block) => {
    const tokens = block.text.split(/\s+/).filter(Boolean);
    if (tokenIndex < 0 || tokenIndex >= tokens.length) return block;
    tokens[tokenIndex] = word;
    const text = tokens.join(" ");
    const words = block.words && block.words.length === tokens.length
      ? block.words.map((w, i) => (i === tokenIndex ? { text: word, support: 1, sources: ["reviewer"], alternatives: {} } : w))
      : block.words;
    return { ...block, text, markdown: markdownFor(block, text), words };
  });
}

/** Undo one automatic correction, restoring the word the model actually read. */
export function undoCorrection(state: DocState, pageIndex: number, blockIndex: number, correction: Correction): DocState {
  return updateBlock(state, pageIndex, blockIndex, (block) => {
    const at = block.text.indexOf(correction.corrected, Math.max(0, correction.position - 2));
    const index = at >= 0 ? at : block.text.indexOf(correction.corrected);
    if (index < 0) return { ...block, corrections: (block.corrections ?? []).filter((c) => c !== correction) };
    const text = block.text.slice(0, index) + correction.original + block.text.slice(index + correction.corrected.length);
    return {
      ...block,
      text,
      markdown: markdownFor(block, text),
      corrections: (block.corrections ?? []).filter((c) => c !== correction),
    };
  });
}

export interface ReviewItem {
  key: string;
  pageIndex: number;
  blockIndex: number;
  tokenIndex?: number;
  kind: "word" | "illegible" | "checkbox" | "table" | "signature" | "other";
  title: string;
  detail: string;
  vote?: WordVote;
}

const KIND_TITLES: Record<string, string> = {
  illegible: "Low-confidence text",
  checkbox: "Checkbox state",
  table: "Table layout",
  signature: "Signature field",
};

/** Everything still waiting for a human: flagged blocks and disputed words. */
export function reviewItems(state: DocState): ReviewItem[] {
  const items: ReviewItem[] = [];
  state.pages.forEach((page, p) => page.blocks.forEach((block, b) => {
    if (block.review_required && !state.resolved.includes(`${p}:${b}`)) {
      items.push({
        key: `${p}:${b}`,
        pageIndex: p,
        blockIndex: b,
        kind: (["illegible", "checkbox", "table", "signature"].includes(block.kind) ? block.kind : "other") as ReviewItem["kind"],
        title: KIND_TITLES[block.kind] ?? "Check this line",
        detail: block.kind === "illegible" ? block.raw_text ?? "" : block.kind === "table" ? `${block.cells?.length ?? 0} rows reconstructed` : block.text,
      });
    }
    tokensWithVotes(block).forEach(({ text, vote }, t) => {
      if (vote && isDisputed(vote) && !state.resolved.includes(`${p}:${b}:${t}`)) {
        items.push({ key: `${p}:${b}:${t}`, pageIndex: p, blockIndex: b, tokenIndex: t, kind: "word", title: "Models disagreed", detail: text, vote });
      }
    });
  }));
  return items;
}

export function resolve(state: DocState, key: string): DocState {
  return state.resolved.includes(key) ? state : { ...state, resolved: [...state.resolved, key] };
}

/** Share of a block's words that won a majority of the vote (null without votes). */
export function agreement(block: StructuredBlock): number | null {
  const votes = block.words?.filter((w) => w.support !== null) ?? [];
  if (!votes.length) return null;
  return votes.filter((w) => !isDisputed(w)).length / votes.length;
}

export function allCorrections(state: DocState) {
  return state.pages.flatMap((page, p) => page.blocks.flatMap((block, b) => (block.corrections ?? []).map((correction) => ({ pageIndex: p, blockIndex: b, correction }))));
}
