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

/** Map a 0..1 confidence to a heat colour (red → amber → green). */
export function confidenceColor(confidence: number, alpha = 1) {
  const c = Math.max(0, Math.min(1, confidence));
  const hue = c < 0.5 ? 8 + c * 2 * 32 : 40 + (c - 0.5) * 2 * 105;
  return `hsla(${hue.toFixed(0)}, 92%, 58%, ${alpha})`;
}
