import { describe, expect, it } from "vitest";
import { cer, charDiff, editDistance, plainText, wer } from "./text";

describe("text metrics", () => {
  it("computes edit distance", () => {
    expect(editDistance([..."kitten"], [..."sitting"])).toBe(3);
    expect(editDistance([], [..."abc"])).toBe(3);
  });

  it("computes CER and WER against a reference", () => {
    expect(cer("hello world", "hello world")).toBe(0);
    expect(cer("helo world", "hello world")).toBeCloseTo(1 / 11);
    expect(wer("the cat sat", "the cat sat down")).toBeCloseTo(0.25);
    expect(cer("", "")).toBe(0);
  });

  it("produces a character diff that reconstructs both strings", () => {
    const ops = charDiff("helo wurld", "hello world");
    const prediction = ops.filter((op) => op.type !== "delete").map((op) => op.text).join("");
    const reference = ops.filter((op) => op.type !== "insert").map((op) => op.text).join("");
    expect(prediction).toBe("helo wurld");
    expect(reference).toBe("hello world");
  });

  it("strips markdown structure for plain-text export", () => {
    expect(plainText("## Page 1\n\n## Title\n| a | b |\n| --- | --- |\n| 1 | 2 |")).not.toContain("|");
  });
});
