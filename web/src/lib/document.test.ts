import { describe, expect, it } from "vitest";
import type { OcrResponse, StructuredBlock } from "../types";
import {
  agreement, docMarkdown, initDoc, isDisputed, replaceWord, resolve, reviewItems, setBlockText, tokensWithVotes, undoCorrection,
} from "./document";

const block = (patch: Partial<StructuredBlock>): StructuredBlock => ({
  kind: "text", text: "", markdown: "", polygon: [], confidence: 0.9, review_required: false, ...patch,
});

const result = (blocks: StructuredBlock[]): OcrResponse => ({
  filename: "x.png", model: "m", language: "en", inference_ms: 1, text: "", markdown: "",
  pages: [{ page_number: 1, text: "", markdown: "", width: 10, height: 10, preview_image: "", blocks, regions: [] }],
});

const word = (text: string, support: number, alternatives: Record<string, number> = {}) => ({ text, support, sources: ["a"], alternatives });

describe("document model", () => {
  it("derives Markdown from blocks and lets a raw edit take over", () => {
    const doc = initDoc(result([block({ kind: "heading", text: "Title", markdown: "## Title" }), block({ text: "Body", markdown: "Body" })]));
    expect(docMarkdown(doc)).toBe("## Page 1\n\n## Title\nBody");
    expect(docMarkdown({ ...doc, markdownOverride: "edited" })).toBe("edited");
  });

  it("keeps a heading's level when its text is edited", () => {
    const doc = setBlockText(initDoc(result([block({ kind: "heading", text: "Title", markdown: "### Title" })])), 0, 0, "New title");
    expect(doc.pages[0].blocks[0].markdown).toBe("### New title");
  });

  it("turns an edited illegible line into reviewed text", () => {
    const doc = setBlockText(initDoc(result([block({ kind: "illegible", text: "", raw_text: "smudge", markdown: "[Illegible]", review_required: true })])), 0, 0, "smudged");
    expect(doc.pages[0].blocks[0]).toMatchObject({ kind: "text", text: "smudged", review_required: false });
    expect(reviewItems(doc)).toHaveLength(0);
  });

  it("flags disputed words and settles them when an alternative is picked", () => {
    const words = [word("Tin", 1), word("fan", 0.34, { Pan: 0.66 }), word("Alley", 1)];
    let doc = initDoc(result([block({ text: "Tin fan Alley", markdown: "Tin fan Alley", words })]));
    expect(isDisputed(words[1])).toBe(true);
    expect(agreement(doc.pages[0].blocks[0])).toBeCloseTo(2 / 3);
    expect(reviewItems(doc).map((item) => item.key)).toEqual(["0:0:1"]);

    doc = replaceWord(doc, 0, 0, 1, "Pan");
    expect(doc.pages[0].blocks[0].text).toBe("Tin Pan Alley");
    expect(tokensWithVotes(doc.pages[0].blocks[0])[1].vote?.sources).toEqual(["reviewer"]);
    expect(agreement(doc.pages[0].blocks[0])).toBe(1);
    expect(reviewItems(doc)).toHaveLength(0);
  });

  it("does not pair votes with tokens when the counts differ", () => {
    const tokens = tokensWithVotes(block({ text: "one two", words: [word("one", 0.2)] }));
    expect(tokens.every((token) => token.vote === null)).toBe(true);
  });

  it("restores the original word when an automatic fix is undone", () => {
    const correction = { original: "claer", corrected: "clear", kind: "spelling", position: 2 };
    let doc = initDoc(result([block({ text: "A clear result", markdown: "A clear result", corrections: [correction] })]));
    doc = undoCorrection(doc, 0, 0, doc.pages[0].blocks[0].corrections![0]);
    expect(doc.pages[0].blocks[0].text).toBe("A claer result");
    expect(doc.pages[0].blocks[0].corrections).toHaveLength(0);
  });

  it("removes resolved items from the review queue", () => {
    const doc = initDoc(result([block({ kind: "checkbox", text: "[x] Yes", markdown: "[x] Yes", review_required: true })]));
    expect(reviewItems(doc)).toHaveLength(1);
    expect(reviewItems(resolve(doc, "0:0"))).toHaveLength(0);
  });
});
