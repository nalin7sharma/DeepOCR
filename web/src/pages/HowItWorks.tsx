import { ArrowRight, FileInput, GitCompareArrows, Layers, ListChecks, ScanLine, ScanText, SpellCheck2, Wand2 } from "lucide-react";

const STAGES = [
  {
    icon: FileInput,
    title: "Read the file",
    body: "PDFs are rendered page by page at twice their native resolution (up to 100 pages); images are decoded as they are. Nothing is stored: each page lives in memory only while it is processed.",
    code: "api/documents.py · decode_document()",
  },
  {
    icon: Wand2,
    title: "Clean up the image",
    body: "The page is straightened by estimating the skew of its text lines, speckle noise is removed, and CLAHE lifts faint ink. The Photo preset also evens out shadows with adaptive thresholding.",
    code: "api/preprocess.py · estimate_skew(), preprocess_page()",
  },
  {
    icon: ScanLine,
    title: "Find the text lines",
    body: "DBNet, a segmentation network, marks every pixel that belongs to text and turns those regions into line polygons, even when lines are tilted or close together.",
    code: "PaddleOCR detector · det_db_box_thresh = 0.5",
  },
  {
    icon: ScanText,
    title: "Read each line",
    body: "Each line is cropped, normalised to the model’s input size and read: by our CNN-BiLSTM or CRNN (CTC decoding), by TrOCR (a Vision Transformer with an autoregressive decoder), or by all five models at once.",
    code: "api/engines.py · read_in_chunks()",
  },
  {
    icon: GitCompareArrows,
    title: "Vote on every word",
    body: "With the consensus engine, the five readings of a line are aligned word by word and each word is decided by a weighted vote (model weight × confidence). Words that win no majority are flagged so a person can check them.",
    code: "postprocessing/ensemble.py · rover()",
    isNew: true,
  },
  {
    icon: Layers,
    title: "Rebuild the layout",
    body: "Lines are grouped into reading order and classified as headings, paragraphs, key-value rows, ruled tables, checkboxes and signature fields, then written out as Markdown.",
    code: "api/structure.py · reconstruct_page()",
  },
  {
    icon: SpellCheck2,
    title: "Correct OCR errors",
    body: "A dictionary and character-confusion model fixes typical misreadings such as 0↔O, 1↔l and 5↔S, while leaving names alone. Every change is listed in the Studio and can be undone.",
    code: "postprocessing/text_corrector.py · correct_with_report()",
  },
  {
    icon: ListChecks,
    title: "Review and export",
    body: "The Studio shows each line on the page with its confidence or agreement, collects everything that needs a human check into one queue, and exports Markdown, plain text or JSON with coordinates.",
    code: "web/src/pages/Studio.tsx",
  },
];

export function HowItWorks({ onTry }: { onTry: () => void }) {
  return (
    <div className="page">
      <div className="ph" style={{ maxWidth: 920, margin: "0 auto" }}>
        <h1>How DeepOCR reads a document</h1>
        <p>Eight stages turn a scan or photo into structured, editable text. The Studio shows each one live while your document is processed.</p>
      </div>
      <ol className="timeline">
        {STAGES.map(({ icon: Icon, title, body, code, isNew }, index) => (
          <li key={title} className={`card tl ${isNew ? "new" : ""}`}>
            <span className="tl-icon"><Icon size={22} /></span>
            <div>
              <h3>{title}{isNew && <span className="badge accent" style={{ marginLeft: 10, verticalAlign: 2 }}>Consensus</span>}</h3>
              <p>{body}</p>
              <code>{code}</code>
            </div>
            <span className="tl-n">{String(index + 1).padStart(2, "0")}</span>
          </li>
        ))}
      </ol>
      <div style={{ display: "flex", justifyContent: "center", marginTop: 28 }}>
        <button type="button" className="btn primary lg" onClick={onTry}>Try it on a document<ArrowRight size={17} /></button>
      </div>
    </div>
  );
}
