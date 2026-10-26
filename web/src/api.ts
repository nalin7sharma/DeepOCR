import type { EnginesResponse, MetricsResponse, OcrOptions, OcrResponse, SampleDoc } from "./types";

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
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

/** POST a document for OCR. Retries automatically while the single worker is busy (HTTP 429). */
export async function recognizeDocument(
  file: File,
  options?: Partial<OcrOptions>,
  signal?: AbortSignal,
  onBusy?: (attempt: number) => void,
): Promise<OcrResponse> {
  for (let attempt = 0; ; attempt += 1) {
    const body = new FormData();
    body.append("file", file);
    if (options?.engine) body.append("engine", options.engine);
    if (options?.preprocess) body.append("preprocess", options.preprocess.join(",") || "none");
    if (options?.correct !== undefined) body.append("correct", String(options.correct));
    const response = await fetch("/api/v1/ocr", { method: "POST", body, signal });
    if (response.status === 429 && attempt < 20) {
      onBusy?.(attempt + 1);
      const retry = Number(response.headers.get("Retry-After")) || 3;
      await sleep(retry * 1000, signal);
      continue;
    }
    if (!response.ok) throw await readError(response);
    return response.json() as Promise<OcrResponse>;
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
  return new File([blob], sample.file.split("/").pop() || sample.id, { type: blob.type });
}
