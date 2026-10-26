export interface WordVote {
  text: string;
  /** Share of the total vote this word won (0..1); null when not from the consensus engine. */
  support: number | null;
  sources: string[];
  alternatives: Record<string, number>;
}

export interface TextRegion {
  polygon: number[][];
  text: string;
  confidence: number;
  review_required: boolean;
  words?: WordVote[] | null;
}

export interface Correction {
  original: string;
  corrected: string;
  kind: "spelling" | "confusion" | string;
  position: number;
}

export type BlockKind = "text" | "title" | "heading" | "table" | "checkbox" | "signature" | "illegible";

export interface StructuredBlock {
  kind: BlockKind | string;
  text: string;
  markdown: string;
  polygon: number[][];
  confidence: number;
  review_required: boolean;
  raw_text?: string | null;
  cells?: string[][] | null;
  checkbox_state?: string | null;
  original_text?: string | null;
  corrections?: Correction[];
  words?: WordVote[] | null;
}

export interface PreprocessStage {
  id: string;
  label: string;
  ms: number;
  image: string;
}

export interface OcrPage {
  page_number: number;
  text: string;
  markdown: string;
  width: number;
  height: number;
  preview_image: string;
  blocks: StructuredBlock[];
  regions: TextRegion[];
  preprocessing?: { applied: string[]; deskew_angle: number | null; stages: PreprocessStage[] };
  timings?: Record<string, number>;
}

export interface OcrStats {
  pages: number;
  lines: number;
  words: number;
  characters: number;
  regions: number;
  mean_confidence: number;
  low_confidence: number;
  review_items: number;
  corrections: number;
  confidence_histogram: number[];
}

export interface OcrResponse {
  filename: string;
  model: string;
  engine?: string;
  language: string;
  inference_ms: number;
  text: string;
  markdown: string;
  pages: OcrPage[];
  timings?: Record<string, number>;
  stats?: OcrStats | null;
  options?: { engine?: string; preprocess?: string[]; correct?: boolean };
}

export interface JobSnapshot {
  id: string;
  filename: string;
  engine: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  stage: string;
  stage_label: string;
  page: number;
  pages: number;
  lines_done: number;
  lines_total: number;
  members: Record<string, { done: number; total: number }>;
  stage_times: Record<string, number>;
  elapsed_ms: number;
  error?: string | null;
  error_status?: number | null;
  result?: OcrResponse;
}

export interface EvalSetMetrics {
  cer: number;
  wer: number;
  accuracy: number;
  cer_raw?: number;
  ms_per_line: number;
  samples: number;
}

export interface EngineInfo {
  id: string;
  label: string;
  family: string;
  description: string;
  detector: string;
  recognizer: string;
  available: boolean;
  loaded: boolean;
  load_ms?: number | null;
  error?: string | null;
  info?: {
    architecture?: string;
    parameters?: number | null;
    input?: string;
    epoch?: number | null;
    val_cer?: number | null;
    datasets?: string[] | null;
    training?: boolean;
  };
  metrics?: Record<string, EvalSetMetrics> | null;
}

export interface EnginesResponse {
  default: string;
  busy: boolean;
  engines: EngineInfo[];
  preprocessing_steps: string[];
}

export interface TrainingRun {
  model: string;
  epoch: number;
  epochs_planned: number;
  finished: boolean;
  best_val_cer: number | null;
  epoch_seconds: number[];
  elapsed_seconds: number;
  meta: Record<string, unknown>;
  history: {
    train_loss: (number | null)[];
    val_loss?: number[];
    train_cer?: number[];
    val_cer: number[];
    val_wer?: number[];
    lr?: number[];
    step?: number[];
    val_cer_iam?: number[];
    val_cer_synthetic?: number[];
  };
}

export interface EvaluationSummary {
  generated_at?: string;
  test_sets?: Record<string, { label: string; samples: number; description?: string }>;
  models: Record<string, Record<string, EvalSetMetrics> & { _info?: Record<string, unknown> }>;
  notes?: string[];
}

export interface SamplePrediction {
  set: string;
  image: string;
  ground_truth: string;
  predictions: Record<string, { text: string; cer: number }>;
}

export interface RobustnessResult {
  conditions: { id: string; label: string; levels: (number | string)[] }[];
  models: Record<string, Record<string, number[]>>;
  samples: number;
}

export interface EnsembleReport {
  method: string;
  lines: number;
  weights: Record<string, number>;
  sets: Record<string, Record<string, { cer: number; wer: number; accuracy: number; samples: number }>>;
}

export interface MetricsResponse {
  evaluation: EvaluationSummary | null;
  samples: SamplePrediction[] | null;
  robustness: RobustnessResult | null;
  ensemble: EnsembleReport | null;
  training: Record<string, TrainingRun>;
}

export interface SampleDoc {
  id: string;
  file: string;
  thumb?: string;
  title: string;
  subtitle?: string;
  kind?: string;
  suggested_engine?: string;
  preprocess?: string[];
  /** Exact text of the sample, used to score engines on the Compare page. */
  ground_truth?: string;
}

export interface OcrOptions {
  engine: string;
  preprocess: string[];
  correct: boolean;
}
