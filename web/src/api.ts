import type { EnginesResponse, JobSnapshot, MetricsResponse, OcrOptions, OcrResponse, SampleDoc } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function readError(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status}).`;
  try {
    const payload: { detail?: unknown } = await response.json();
    if (typeof payload.detail === "string") message = payload.detail;
    else if (Array.isArray(payload.detail)) message = "The request was not valid.";
  } catch {
    // Keep the status-based message when the server did not return JSON.
  }
  return new ApiError(message, response.status);
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });

function formFor(file: File, options?: Partial<OcrOptions>) {
  const body = new FormData();
  body.append("file", file);
  if (options?.engine) body.append("engine", options.engine);
  if (options?.preprocess) body.append("preprocess", options.preprocess.join(",") || "none");
  if (options?.correct !== undefined) body.append("correct", String(options.correct));
  return body;
}

export interface RunCallbacks {
  /** Called with every progress snapshot while the job runs. */
  onProgress?: (job: JobSnapshot) => void;
  /** Called while waiting because the single worker is busy with another document. */
  onBusy?: (attempt: number) => void;
}

/**
 * Run OCR as a background job and poll its live progress (stage, page, lines read per model).
 * Retries automatically while another document is being processed (HTTP 429).
 * Aborting the signal cancels the job on the server.
 */
export async function runOcr(
  file: File,
  options?: Partial<OcrOptions>,
  signal?: AbortSignal,
  callbacks: RunCallbacks = {},
): Promise<OcrResponse> {
  let job: JobSnapshot | null = null;
  for (let attempt = 0; ; attempt += 1) {
    const response = await fetch("/api/v1/jobs", { method: "POST", body: formFor(file, options), signal });
    if (response.status === 429 && attempt < 40) {
      callbacks.onBusy?.(attempt + 1);
      await sleep((Number(response.headers.get("Retry-After")) || 3) * 1000, signal);
      continue;
    }
    if (!response.ok) throw await readError(response);
    job = (await response.json()) as JobSnapshot;
    break;
  }
  const id = job.id;
  const cancel = () => {
    void fetch(`/api/v1/jobs/${id}`, { method: "DELETE" }).catch(() => undefined);
  };
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    for (;;) {
      await sleep(350, signal);
      const response = await fetch(`/api/v1/jobs/${id}`, { signal });
      if (!response.ok) throw await readError(response);
      const snapshot = (await response.json()) as JobSnapshot;
      callbacks.onProgress?.(snapshot);
      if (snapshot.status === "done" && snapshot.result) return snapshot.result;
      if (snapshot.status === "error") throw new ApiError(snapshot.error || "OCR failed.", snapshot.error_status || 500);
      if (snapshot.status === "cancelled") throw new DOMException("Aborted", "AbortError");
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw await readError(response);
  return response.json() as Promise<T>;
}

export const fetchEngines = (signal?: AbortSignal) => getJson<EnginesResponse>("/api/v1/engines", signal);
export const fetchMetrics = (signal?: AbortSignal) => getJson<MetricsResponse>("/api/v1/metrics", signal);
export const fetchSamples = (signal?: AbortSignal) => getJson<SampleDoc[]>("/api/v1/samples", signal);

export async function fetchSampleFile(sample: SampleDoc): Promise<File> {
  const response = await fetch(`/api/v1/samples/${encodeURIComponent(sample.id)}`);
  if (!response.ok) throw await readError(response);
  const blob = await response.blob();
  const type = blob.type || (sample.file.endsWith(".pdf") ? "application/pdf" : "image/png");
  return new File([blob], sample.file.split("/").pop() || sample.id, { type });
}

export const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError";
