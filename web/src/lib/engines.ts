import type { EngineInfo } from "../types";

/** Display order: most accurate first. */
export const ENGINE_ORDER = ["ensemble", "trocr_ft", "crnn", "cnn_bilstm", "trocr", "paddle"];

export const ENGINE_COLORS: Record<string, string> = {
  ensemble: "#4f46e5",
  trocr_ft: "#f79009",
  trocr: "#16b364",
  crnn: "#7a5af8",
  cnn_bilstm: "#0ba5ec",
  paddle: "#667085",
  cnn_bilstm_pp: "#7cd4fd",
  crnn_pp: "#bdb4fe",
};

export const ENGINE_SHORT: Record<string, string> = {
  ensemble: "Consensus · 5 models",
  trocr_ft: "TrOCR · fine-tuned",
  trocr: "TrOCR",
  crnn: "CRNN",
  cnn_bilstm: "CNN-BiLSTM",
  paddle: "PaddleOCR",
};

/** Short provenance tag and one-line description shown in the engine list. */
export const ENGINE_COPY: Record<string, { tag: string; ours: boolean; desc: string }> = {
  ensemble: { tag: "Ours · ensemble", ours: true, desc: "All models read each line, then vote word by word" },
  trocr_ft: { tag: "Ours · fine-tuned", ours: true, desc: "Vision Transformer tuned on our data" },
  crnn: { tag: "Ours · trained", ours: true, desc: "CNN + BiLSTM + CTC, 2.8 M parameters" },
  cnn_bilstm: { tag: "Ours · baseline", ours: true, desc: "CNN features + BiLSTM + CTC, 2.4 M" },
  trocr: { tag: "Pretrained", ours: false, desc: "Microsoft TrOCR-small, as released" },
  paddle: { tag: "Reference", ours: false, desc: "Industrial PP-OCR, strong on print" },
};

export const colorFor = (id: string, index = 0) =>
  ENGINE_COLORS[id] ?? ["#ee46bc", "#06aed4", "#66c61c", "#f04438"][index % 4];

export const shortName = (id: string, fallback?: string) => ENGINE_SHORT[id] ?? fallback ?? id;

export function sortEngines(engines: EngineInfo[]): EngineInfo[] {
  return [...engines].sort((a, b) => rank(a.id) - rank(b.id));
}

const rank = (id: string) => {
  const index = ENGINE_ORDER.indexOf(id);
  return index === -1 ? 99 : index;
};

/** Mean measured inference time per line across the test sets, or null. */
export function msPerLine(engine: EngineInfo): number | null {
  const sets = Object.values(engine.metrics ?? {}).filter((m) => m && typeof m.ms_per_line === "number");
  if (!sets.length) return null;
  return sets.reduce((sum, m) => sum + m.ms_per_line, 0) / sets.length;
}

export function speedLabel(ms: number | null): string {
  if (ms === null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 100 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export const ENGINE_BADGE: Record<string, { text: string; tone: "success" | "accent" }> = {
  ensemble: { text: "Most accurate", tone: "success" },
  trocr_ft: { text: "Best balance", tone: "accent" },
};
