import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { EnginesResponse, JobSnapshot, MetricsResponse, OcrResponse, SampleDoc } from "./types";

Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:test-preview") });
Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });

const engines: EnginesResponse = {
  default: "paddle",
  busy: false,
  preprocessing_steps: ["deskew", "denoise", "contrast", "binarize"],
  engines: [
    { id: "paddle", label: "PaddleOCR", family: "Pretrained reference", description: "PP-OCR", detector: "DBNet", recognizer: "PP-OCR", available: true, loaded: true, info: {}, metrics: null },
    { id: "crnn", label: "DeepOCR · CRNN", family: "Ours", description: "CRNN", detector: "DBNet", recognizer: "CRNN + CTC", available: true, loaded: false, info: { parameters: 2_780_115 }, metrics: { iam: { cer: 12.5, wer: 30, accuracy: 20, ms_per_line: 4, samples: 600 } } },
    { id: "trocr", label: "TrOCR Transformer", family: "Transfer", description: "TrOCR", detector: "DBNet", recognizer: "ViT", available: false, loaded: false, error: "missing", info: {}, metrics: null },
  ],
};

const square = (y: number) => [[20, y], [300, y], [300, y + 20], [20, y + 20]];

const ocrResult: OcrResponse = {
  filename: "note.png",
  model: "PaddleOCR",
  engine: "paddle",
  language: "en",
  inference_ms: 830,
  text: "",
  markdown: "",
  timings: { decode: 10, preprocess: 40, recognition: 700, layout: 20, postprocess: 5, total: 830 },
  stats: { pages: 1, lines: 4, words: 7, characters: 40, regions: 4, mean_confidence: 0.81, low_confidence: 1, review_items: 2, corrections: 1, confidence_histogram: [0, 0, 1, 0, 0, 0, 0, 0, 1, 2] },
  options: { engine: "paddle", preprocess: ["deskew"], correct: true },
  pages: [{
    page_number: 1,
    text: "",
    markdown: "## Big Title\nA clear result\nSignature: [Signature]\n[Illegible]",
    width: 640,
    height: 320,
    preview_image: "dGVzdA==",
    preprocessing: { applied: ["deskew"], deskew_angle: 1.5, stages: [
      { id: "original", label: "Original", ms: 0, image: "b3JpZw==" },
      { id: "deskew", label: "Deskewed (+1.50°)", ms: 12, image: "ZGVz" },
    ] },
    blocks: [
      { kind: "heading", text: "Big Title", markdown: "## Big Title", polygon: square(5), confidence: 0.97, review_required: false },
      { kind: "text", text: "A clear result", markdown: "A clear result", polygon: square(30), confidence: 0.93, review_required: false, original_text: "A claer result", corrections: [{ original: "claer", corrected: "clear", kind: "spelling", position: 2 }] },
      { kind: "signature", text: "Signature:", markdown: "Signature: [Signature]", polygon: square(70), confidence: 0.88, review_required: true },
      { kind: "illegible", text: "smudge", raw_text: "smudge", markdown: "[Illegible]", polygon: square(100), confidence: 0.2, review_required: true },
    ],
    regions: [{ polygon: square(30), text: "A clear result", confidence: 0.93, review_required: false }],
  }],
};

const vote = (text: string, support: number, alternatives: Record<string, number> = {}) => ({ text, support, sources: ["trocr_ft"], alternatives });

const consensusResult: OcrResponse = {
  ...ocrResult,
  engine: "ensemble",
  options: { engine: "ensemble", preprocess: [], correct: false },
  pages: [{
    ...ocrResult.pages[0],
    preprocessing: { applied: [], deskew_angle: null, stages: [] },
    blocks: [{
      kind: "text",
      text: "Tin fan Alley",
      markdown: "Tin fan Alley",
      polygon: square(5),
      confidence: 0.8,
      review_required: false,
      words: [vote("Tin", 1), vote("fan", 0.34, { Pan: 0.66 }), vote("Alley", 1)],
    }],
  }],
};

const metrics: MetricsResponse = {
  evaluation: {
    test_sets: { iam: { label: "IAM handwriting", samples: 600 } },
    models: {
      crnn: { iam: { cer: 21.66, wer: 61.47, accuracy: 0.5, ms_per_line: 16, samples: 600 } },
      trocr_ft: { iam: { cer: 6.73, wer: 16.79, accuracy: 30, ms_per_line: 415, samples: 600 } },
      ensemble: { iam: { cer: 5.744, wer: 14.239, accuracy: 42.3, ms_per_line: 960, samples: 600 } },
    },
    notes: [],
  },
  samples: [],
  robustness: null,
  ensemble: {
    method: "ROVER",
    lines: 1200,
    weights: { trocr_ft: 3, crnn: 1 },
    sets: {
      iam: {
        trocr_ft: { cer: 6.73, wer: 16.79, accuracy: 30, samples: 600 },
        crnn: { cer: 21.66, wer: 61.47, accuracy: 0.5, samples: 600 },
        ensemble_cv: { cer: 5.74, wer: 14.24, accuracy: 42.3, samples: 600 },
        oracle_best_line: { cer: 4.27, wer: 12.19, accuracy: 48.8, samples: 600 },
      },
    },
  },
  training: {},
};

const samples: SampleDoc[] = [
  { id: "hw", file: "note.png", title: "Handwritten note", subtitle: "Eight lines", suggested_engine: "paddle", preprocess: [], ground_truth: "Big Title\nA clear result" },
];

let jobCalls: FormData[] = [];

function mockFetch(result: OcrResponse | (() => Response)) {
  jobCalls = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (url.startsWith("/api/v1/engines")) return Response.json(engines);
    if (url.startsWith("/api/v1/metrics")) return Response.json(metrics);
    if (url === "/api/v1/samples") return Response.json(samples);
    if (url.startsWith("/api/v1/samples/")) return new Response(new Blob(["png"], { type: "image/png" }));
    if (url === "/api/v1/jobs" && init?.method === "POST") {
      jobCalls.push(init.body as FormData);
      if (typeof result === "function") return result();
      const engine = String((init.body as FormData).get("engine"));
      const snapshot: Partial<JobSnapshot> = { id: `job${jobCalls.length}`, engine, status: "queued", stage: "queued", members: {}, stage_times: {} };
      return Response.json(snapshot, { status: 202 });
    }
    if (url.startsWith("/api/v1/jobs/")) {
      if (init?.method === "DELETE") return Response.json({ cancelled: true });
      const done: Partial<JobSnapshot> = { id: "job", status: "done", stage: "done", members: {}, stage_times: {}, result: result as OcrResponse };
      return Response.json(done);
    }
    return new Response("not found", { status: 404 });
  }));
}

function upload(file = new File(["image"], "note.png", { type: "image/png" })) {
  fireEvent.change(screen.getByLabelText("Choose document"), { target: { files: [file] } });
}

async function extract() {
  await screen.findByRole("radio", { name: /PaddleOCR/ });
  upload();
  fireEvent.click(await screen.findByRole("button", { name: /Extract text/ }));
  await screen.findByRole("tab", { name: /Review/ }, { timeout: 6000 });
}

async function savedText(): Promise<string> {
  const blob = (URL.createObjectURL as unknown as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as Blob;
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.readAsText(blob);
  });
}

beforeEach(() => {
  window.location.hash = "";
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Studio", () => {
  it("lists engines with their measured error and disables unavailable ones", async () => {
    mockFetch(ocrResult);
    render(<App />);
    expect(await screen.findByRole("radio", { name: /PaddleOCR/ })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /TrOCR/ })).toBeDisabled();
    expect(within(screen.getByRole("radio", { name: /CRNN/ })).getByText("12.5%")).toBeInTheDocument();
  });

  it("runs a job with the chosen engine and options, then opens the review workspace", async () => {
    mockFetch(ocrResult);
    render(<App />);
    fireEvent.click(await screen.findByRole("radio", { name: /CRNN/ }));
    fireEvent.click(screen.getByRole("radio", { name: "Scan" }));
    await extract();

    expect(jobCalls[0].get("engine")).toBe("crnn");
    expect(jobCalls[0].get("preprocess")).toBe("deskew,denoise,contrast");
    expect(jobCalls[0].get("correct")).toBe("true");

    expect(screen.getByRole("heading", { name: "Big Title" })).toBeInTheDocument();
    expect(screen.getByText("✎ Signature field")).toBeInTheDocument();
    expect(screen.getByText("Illegible")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Review\s*2/ })).toBeInTheDocument();
    expect(screen.getByText("Straightened 1.5°")).toBeInTheDocument();
  });

  it("undoes an automatic fix and exports the edited lines", async () => {
    mockFetch(ocrResult);
    render(<App />);
    await extract();

    const fixes = screen.getByRole("region", { name: "Automatic fixes and review" });
    fireEvent.click(within(fixes).getByRole("button", { name: /Undo/ }));
    fireEvent.click(screen.getByRole("tab", { name: /Lines/ }));
    expect(screen.getByText("claer", { exact: false })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Edit line 1" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Edit line 1" }), { target: { value: "Better Title" } });
    fireEvent.click(screen.getByRole("button", { name: /Save/ }));

    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    fireEvent.click(screen.getByRole("button", { name: "Markdown" }));
    const text = await savedText();
    expect(text).toContain("## Better Title");
    expect(text).toContain("A claer result");
  });

  it("lets the reviewer swap a disputed word for another model's reading", async () => {
    mockFetch(consensusResult);
    render(<App />);
    await extract();

    expect(screen.getByRole("radio", { name: /Agreement/ })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("button", { name: /fan: models disagreed/ }));
    const popover = screen.getByRole("dialog", { name: /Alternatives for fan/ });
    fireEvent.click(within(popover).getByRole("button", { name: /Pan/ }));
    expect(screen.getByText("Pan")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Review\s*0/ })).toBeInTheDocument();
  });

  it("switches overlays on the page", async () => {
    mockFetch(ocrResult);
    render(<App />);
    await extract();
    expect(screen.getByRole("button", { name: /Heading 1: Big Title/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /Confidence/ }));
    expect(screen.queryByRole("button", { name: /Heading 1/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Before \/ after/ })).toBeInTheDocument();
  });

  it("rejects unsupported and oversized files before uploading", async () => {
    mockFetch(ocrResult);
    render(<App />);
    await screen.findByRole("radio", { name: /PaddleOCR/ });
    upload(new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported file type");
    const big = new File(["x"], "huge.png", { type: "image/png" });
    Object.defineProperty(big, "size", { value: 101 * 1024 * 1024 });
    upload(big);
    expect(await screen.findByRole("alert")).toHaveTextContent("100 MB");
    expect(jobCalls).toHaveLength(0);
  });

  it("surfaces server errors", async () => {
    mockFetch(() => Response.json({ detail: "The PDF file is invalid or unsupported." }, { status: 400 }));
    render(<App />);
    upload();
    fireEvent.click(await screen.findByRole("button", { name: /Extract text/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The PDF file is invalid");
  });
});

describe("Compare", () => {
  it("scores each engine against a sample's ground truth", async () => {
    mockFetch(ocrResult);
    window.location.hash = "#/compare";
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Handwritten note/ }));
    expect(await screen.findByText("Most accurate", {}, { timeout: 4000 })).toBeInTheDocument();
    expect(jobCalls.map((body) => body.get("engine"))).toEqual(["crnn", "paddle"]);
    expect(jobCalls[0].get("correct")).toBe("false");
    expect(screen.getAllByText("CER").length).toBe(2);
  });
});

describe("Research", () => {
  it("shows the measured results and the consensus gain", async () => {
    mockFetch(ocrResult);
    window.location.hash = "#/research";
    render(<App />);
    expect((await screen.findAllByText("5.74%")).length).toBeGreaterThan(0);
    expect(screen.getByText("Consensus vote (5 models)")).toBeInTheDocument();
    expect(screen.getByText("Fewer errors with voting")).toBeInTheDocument();
  });
});

describe("How it works", () => {
  it("includes the voting stage", async () => {
    mockFetch(ocrResult);
    window.location.hash = "#/how-it-works";
    render(<App />);
    expect(await screen.findByRole("heading", { name: /Vote on every word/ })).toBeInTheDocument();
  });
});
