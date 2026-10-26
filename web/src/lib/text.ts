/** Levenshtein distance over arbitrary token arrays (two-row DP). */
export function editDistance<T>(a: T[], b: T[]): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const current = [i];
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = a[i - 1] === b[j - 1]
        ? previous[j - 1]
        : 1 + Math.min(previous[j - 1], previous[j], current[j - 1]);
    }
    previous = current;
  }
  return previous[b.length];
}

const normalize = (text: string) => text.replace(/\s+/g, " ").trim();

export function cer(prediction: string, reference: string): number {
  const ref = [...normalize(reference)];
  if (!ref.length) return prediction.trim() ? 1 : 0;
  return editDistance([...normalize(prediction)], ref) / ref.length;
}

export function wer(prediction: string, reference: string): number {
  const ref = normalize(reference).split(" ").filter(Boolean);
  if (!ref.length) return prediction.trim() ? 1 : 0;
  return editDistance(normalize(prediction).split(" ").filter(Boolean), ref) / ref.length;
}

export type DiffOp = { type: "equal" | "insert" | "delete"; text: string };

/** Character-level diff (prediction vs reference) via an LCS table; inputs are line-sized. */
export function charDiff(prediction: string, reference: string): DiffOp[] {
  const a = [...prediction];
  const b = [...reference];
  if (a.length * b.length > 250_000) {
    return [{ type: "equal", text: prediction }];
  }
  const table = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const ops: DiffOp[] = [];
  const push = (type: DiffOp["type"], ch: string) => {
    const last = ops[ops.length - 1];
    if (last && last.type === type) last.text += ch;
    else ops.push({ type, text: ch });
  };
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      push("equal", a[i]);
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      push("insert", a[i]); // extra character in the prediction
      i += 1;
    } else {
      push("delete", b[j]); // character missing from the prediction
      j += 1;
    }
  }
  while (i < a.length) push("insert", a[i++]);
  while (j < b.length) push("delete", b[j++]);
  return ops;
}

export function plainText(markdown: string): string {
  return markdown
    .replace(/^##\s+Page\s+\d+\s*$/gm, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\|\s*-{3}.*$/gm, "")
    .replace(/\|/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export const percent = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(digits)}%`;

export const ms = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
};

export const compact = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
};

export function baseName(filename: string) {
  return filename.replace(/\.[^.]+$/, "") || "document";
}

export function saveFile(name: string, contents: BlobPart, type: string) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

/** Three-step quality scale used for confidence and agreement (0..1). */
export type Level = "hi" | "mid" | "lo";
export const level = (value: number): Level => (value >= 0.85 ? "hi" : value >= 0.6 ? "mid" : "lo");
const LEVEL_RGB: Record<Level, string> = { hi: "23,178,106", mid: "247,144,9", lo: "240,68,56" };
export const levelColor = (value: number, alpha = 1) => `rgba(${LEVEL_RGB[level(value)]},${alpha})`;

export type WordToken = { type: "ok" | "wrong" | "miss"; text: string; expected?: string; br?: boolean };

/**
 * Whole-document word diff of a prediction against the reference text.
 * Aligning the whole document (not line by line) keeps the comparison correct when
 * an engine splits or merges lines differently. Line breaks of the prediction are kept.
 */
export function wordDiff(predictionLines: string[], reference: string): WordToken[] {
  const pred: string[] = [];
  const lineOf: number[] = [];
  predictionLines.forEach((line, n) => line.split(/\s+/).filter(Boolean).forEach((w) => { pred.push(w); lineOf.push(n); }));
  const ref = reference.split(/\s+/).filter(Boolean);
  const n = pred.length;
  const m = ref.length;
  if (n * m > 400_000) return pred.map((w, i) => ({ type: "ok", text: w, br: i > 0 && lineOf[i] !== lineOf[i - 1] }));
  // Levenshtein table over words.
  const d = Array.from({ length: n + 1 }, (_, i) => {
    const row = new Uint32Array(m + 1);
    row[0] = i;
    return row;
  });
  for (let j = 0; j <= m; j += 1) d[0][j] = j;
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (pred[i - 1] === ref[j - 1] ? 0 : 1));
    }
  }
  const ops: { i: number | null; j: number | null }[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (pred[i - 1] === ref[j - 1] ? 0 : 1)) {
      ops.push({ i: i - 1, j: j - 1 });
      i -= 1;
      j -= 1;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      ops.push({ i: i - 1, j: null });
      i -= 1;
    } else {
      ops.push({ i: null, j: j - 1 });
      j -= 1;
    }
  }
  ops.reverse();
  const tokens: WordToken[] = [];
  let lastLine = 0;
  for (const op of ops) {
    if (op.i === null) {
      tokens.push({ type: "miss", text: "‸", expected: ref[op.j as number] });
      continue;
    }
    const br = tokens.length > 0 && lineOf[op.i] !== lastLine;
    lastLine = lineOf[op.i];
    if (op.j !== null && pred[op.i] === ref[op.j]) tokens.push({ type: "ok", text: pred[op.i], br });
    else tokens.push({ type: "wrong", text: pred[op.i], expected: op.j === null ? undefined : ref[op.j], br });
  }
  // Merge consecutive "missing" markers into one caret listing every missing word.
  return tokens.reduce<WordToken[]>((out, token) => {
    const last = out[out.length - 1];
    if (token.type === "miss" && last?.type === "miss") last.expected = `${last.expected} ${token.expected}`;
    else out.push(token);
    return out;
  }, []);
}
