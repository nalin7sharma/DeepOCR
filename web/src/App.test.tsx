import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { EnginesResponse, MetricsResponse, OcrResponse } from "./types";

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

const ocrResult: OcrResponse = {
  filename: "note.png",
  model: "PaddleOCR",
  engine: "paddle",
  language: "en",
  inference_ms: 830,
  text: "## Page 1\n\n## Big Title\nA clear result\nSignature: [Signature]\n[Illegible]",
  markdown: "## Page 1\n\n## Big Title\nA clear result\nSignature: [Signature]\n[Illegible]",
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
      { kind: "heading", text: "Big Title", markdown: "## Big Title", polygon: [[20, 5], [300, 5], [300, 25], [20, 25]], confidence: 0.97, review_required: false },
      { kind: "text", text: "A clear result", markdown: "A clear result", polygon: [[20, 30], [190, 30], [190, 62], [20, 62]], confidence: 0.93, review_required: false, original_text: "A claer result", corrections: [{ original: "claer", corrected: "clear", kind: "spelling", position: 2 }] },
      { kind: "signature", text: "Signature:", markdown: "Signature: [Signature]", polygon: [[20, 70], [190, 70], [190, 95], [20, 95]], confidence: 0.88, review_required: true },
      { kind: "illegible", text: "smudge", raw_text: "smudge", markdown: "[Illegible]", polygon: [[20, 100], [190, 100], [190, 120], [20, 120]], confidence: 0.2, review_required: true },
    ],
    regions: [
      { polygon: [[20, 30], [190, 30], [190, 62], [20, 62]], text: "A clear result", confidence: 0.93, review_required: false },
    ],
  }],
};

const metrics: MetricsResponse = {
  evaluation: {
    test_sets: { iam: { label: "IAM handwriting", samples: 600 } },
    models: {
      crnn: { iam: { cer: 11.25, wer: 31.5, accuracy: 18, ms_per_line: 3.2, samples: 600 } },
      trocr: { iam: { cer: 4.1, wer: 11.2, accuracy: 50, ms_per_line: 410, samples: 600 } },
    },
    notes: ["Held-out test split."],
  },
  samples: [],
  robustness: null,
  training: {},
};

let ocrCalls: FormData[] = [];

function mockFetch(ocr: () => Response | Promise<Response>) {
  ocrCalls = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (url.startsWith("/api/v1/engines")) return Response.json(engines);
    if (url.startsWith("/api/v1/metrics")) return Response.json(metrics);
    if (url.startsWith("/api/v1/samples")) return Response.json([]);
    if (url.startsWith("/api/v1/ocr")) {
      ocrCalls.push(init?.body as FormData);
      return ocr();
    }
    return new Response("not found", { status: 404 });
  }));
}

function upload(file = new File(["image"], "note.png", { type: "image/png" })) {
  fireEvent.change(screen.getByLabelText("Choose document"), { target: { files: [file] } });
}

beforeEach(() => {
  window.location.hash = "";
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DeepOCR Studio", () => {
  it("lists engines and disables unavailable ones", async () => {
    mockFetch(() => Response.json(ocrResult));
    render(<App />);
    expect(await screen.findByRole("radio", { name: /PaddleOCR/ })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /TrOCR/ })).toBeDisabled();
    expect(screen.getByText("handwriting CER 12.5%")).toBeInTheDocument();
  });

  it("runs OCR with the chosen engine and options, then shows the review workspace", async () => {
    mockFetch(() => Response.json(ocrResult));
    render(<App />);
    fireEvent.click(await screen.findByRole("radio", { name: /CRNN/ }));
    upload();
    fireEvent.click(screen.getByRole("button", { name: /Denoise/ }));
    fireEvent.click(screen.getByRole("button", { name: /Extract text/ }));

    expect(await screen.findByText("note.png")).toBeInTheDocument();
    const body = ocrCalls[0];
    expect(body.get("engine")).toBe("crnn");
    expect(body.get("preprocess")).toBe("deskew,denoise");
    expect(body.get("correct")).toBe("true");

    expect(screen.getByText("Pipeline complete")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Big Title" })).toBeInTheDocument();
    expect(screen.getByText("✎ Signature")).toBeInTheDocument();
    expect(screen.getByText("Illegible")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Review 2/ })).toBeInTheDocument();
  });

  it("shows corrections, editing and exports the edited text", async () => {
    mockFetch(() => Response.json(ocrResult));
    render(<App />);
    upload();
    fireEvent.click(await screen.findByRole("button", { name: /Extract text/ }));
    await screen.findByText("note.png");

    fireEvent.click(screen.getByRole("tab", { name: /Lines/ }));
    expect(await screen.findByText("A claer result")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Editor/ }));
    const editor = await screen.findByLabelText("Editable extracted Markdown");
    fireEvent.change(editor, { target: { value: "Corrected text" } });
    expect(screen.getByText("edited")).toBeInTheDocument();

    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    fireEvent.click(screen.getByRole("button", { name: /Export Markdown/ }));
    expect(click).toHaveBeenCalled();
    const blob = (URL.createObjectURL as unknown as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0] as Blob;
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(blob);
    });
    expect(text).toBe("Corrected text");
  });

  it("rejects unsupported and oversized files before uploading", async () => {
    mockFetch(() => Response.json(ocrResult));
    render(<App />);
    upload(new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported file type");
    const big = new File(["x"], "huge.png", { type: "image/png" });
    Object.defineProperty(big, "size", { value: 101 * 1024 * 1024 });
    upload(big);
    expect(await screen.findByRole("alert")).toHaveTextContent("100 MB");
    expect(ocrCalls).toHaveLength(0);
  });

  it("surfaces server errors", async () => {
    mockFetch(() => Response.json({ detail: "The PDF file is invalid or unsupported." }, { status: 400 }));
    render(<App />);
    upload();
    fireEvent.click(await screen.findByRole("button", { name: /Extract text/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The PDF file is invalid");
  });

  it("switches overlay layers and shows the preprocessing stages", async () => {
    mockFetch(() => Response.json(ocrResult));
    render(<App />);
    upload();
    fireEvent.click(await screen.findByRole("button", { name: /Extract text/ }));
    await screen.findByText("note.png");
    expect(screen.getByRole("button", { name: /Heading 1: Big Title/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: /Confidence/ }));
    expect(screen.queryByRole("button", { name: /Heading 1/ })).not.toBeInTheDocument();
    const stages = screen.getByLabelText("Preprocessing stages");
    expect(within(stages).getByText("Deskewed (+1.50°)")).toBeInTheDocument();
  });
});

describe("Research dashboard", () => {
  it("renders the measured model comparison", async () => {
    mockFetch(() => Response.json(ocrResult));
    window.location.hash = "#/research";
    render(<App />);
    expect((await screen.findAllByText("4.10%")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/TrOCR \(zero-shot\)/).length).toBeGreaterThan(0);
    expect(screen.getByText("Held-out test split.")).toBeInTheDocument();
  });
});
