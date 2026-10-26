import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, ArrowLeft, Check, ChevronDown, Copy, Download, FileImage, FileText, GitCompareArrows, Layers, ListChecks,
  RefreshCw, ScanText, ShieldCheck, Sparkles, Wand2,
} from "lucide-react";
import { fetchSampleFile, isAbort, runOcr } from "../api";
import { Dropzone, validateFile } from "../components/Dropzone";
import { DocumentViewer } from "../components/DocumentViewer";
import { EngineList } from "../components/EngineList";
import { CLEANUP_STEPS, Options, cleanupFromSteps, type Cleanup } from "../components/Options";
import { ProcessingCard } from "../components/ProcessingCard";
import { FixBar, SummaryStrip } from "../components/ResultBits";
import { SampleCards } from "../components/SampleCards";
import { TextPanel, useExports, type DocActions, type Tab } from "../components/TextPanel";
import { Menu, Segmented } from "../components/Ui";
import { useStoredState } from "../hooks";
import {
  docMarkdown, initDoc, replaceWord, resolve, setBlockText, undoCorrection, type DocState,
} from "../lib/document";
import { msPerLine, shortName, sortEngines, speedLabel } from "../lib/engines";
import type { EngineInfo, EnginesResponse, JobSnapshot, OcrResponse, SampleDoc } from "../types";

interface Props {
  engines: EnginesResponse | null;
  samples: SampleDoc[];
  onEnginesChanged: () => void;
}

type Phase = "upload" | "processing" | "results";

interface RunSettings {
  engine: string;
  cleanup: Cleanup;
  correct: boolean;
}

/* ------------------------------------------------------------------ upload: pitch */
function Pitch({ engines }: { engines: EngineInfo[] }) {
  const ensemble = engines.find((engine) => engine.id === "ensemble");
  const printBest = Math.min(...engines.map((engine) => engine.metrics?.synthetic?.cer ?? Infinity));
  const fastest = Math.min(...engines.map((engine) => msPerLine(engine) ?? Infinity));
  const hw = ensemble?.metrics?.iam?.cer;
  return (
    <section className="pitch">
      <span className="badge accent"><Sparkles size={13} />Deep learning OCR for handwriting &amp; complex documents</span>
      <h1>Turn handwriting and messy scans into structured, editable text.</h1>
      <p className="lead">
        Five recognition models read every line and <b>vote on each word</b>. A word several models agree on is far more likely
        to be right. Image cleanup, layout analysis and error correction are included, and every disputed word is flagged for review.
      </p>
      <div className="proof">
        <div><b className="num">{hw !== undefined ? `${hw.toFixed(1)}%` : "5.7%"}</b><span>character error on handwriting from unseen writers</span></div>
        <div><b className="num">{Number.isFinite(printBest) ? `${printBest.toFixed(1)}%` : "2.4%"}</b><span>character error on printed &amp; degraded text</span></div>
        <div><b className="num">{Number.isFinite(fastest) ? speedLabel(fastest) : "12 ms"}</b><span>per line with our fastest model, on a laptop CPU</span></div>
      </div>
      <ul className="checks">
        <li><Check />PDF, PNG, JPG, TIFF and WEBP, up to 100 pages</li>
        <li><Check />Headings, tables, checkboxes and signature fields detected</li>
        <li><Check />Export Markdown, plain text or JSON</li>
      </ul>
      <div className="flow card">
        <span className="label">What happens to your document</span>
        <ol>
          <li><span className="flow-icon"><Wand2 size={16} /></span><b>Clean up</b><span>Straighten, denoise, boost contrast</span></li>
          <li><span className="flow-icon"><ScanText size={16} /></span><b>Find &amp; read</b><span>Detect each line; up to five models read it</span></li>
          <li><span className="flow-icon"><GitCompareArrows size={16} /></span><b>Vote</b><span>Words the models agree on win; disagreements are flagged</span></li>
          <li><span className="flow-icon"><Layers size={16} /></span><b>Rebuild layout</b><span>Headings, tables, checkboxes, reading order</span></li>
          <li><span className="flow-icon"><ListChecks size={16} /></span><b>Review</b><span>Fix flagged words, then export</span></li>
        </ol>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ results: rail */
function CleanupNotes({ result, pageIndex }: { result: OcrResponse; pageIndex: number }) {
  const report = result.pages[pageIndex]?.preprocessing;
  const applied = report?.applied ?? [];
  if (!applied.length) return <span className="rail-note"><Check size={14} />Used as uploaded</span>;
  const angle = report?.deskew_angle ?? 0;
  const notes = applied.map((step) => {
    if (step === "deskew") return Math.abs(angle) >= 0.3 ? `Straightened ${Math.abs(angle).toFixed(1)}°` : "Already straight";
    if (step === "denoise") return "Removed speckle noise";
    if (step === "contrast") return "Boosted faint text";
    if (step === "binarize") return "Evened out lighting";
    return step;
  });
  return <>{notes.map((note) => <span key={note} className="rail-note"><Check size={14} />{note}</span>)}</>;
}

function ExportMenu({ result, doc, primary = true }: { result: OcrResponse; doc: DocState; primary?: boolean }) {
  const exports = useExports(result, doc);
  return (
    <Menu
      label="Export"
      trigger={({ open, toggle }) => (
        <button type="button" className={`btn ${primary ? "primary" : ""}`} aria-expanded={open} aria-haspopup="menu" onClick={toggle}>
          <Download size={16} />Export<ChevronDown size={15} />
        </button>
      )}
    >
      {(close) => (
        <>
          <button type="button" role="menuitem" onClick={() => { void exports.copy(); close(); }}><Copy size={15} />Copy Markdown</button>
          <hr />
          <button type="button" role="menuitem" onClick={() => { exports.markdown(); close(); }}><FileText size={15} />Markdown<small>.md</small></button>
          <button type="button" role="menuitem" onClick={() => { exports.text(); close(); }}><FileText size={15} />Plain text<small>.txt</small></button>
          <button type="button" role="menuitem" onClick={() => { exports.json(); close(); }}><FileText size={15} />JSON with boxes<small>.json</small></button>
        </>
      )}
    </Menu>
  );
}

/* ------------------------------------------------------------------ page */
export function Studio({ engines, samples, onEnginesChanged }: Props) {
  const [phase, setPhase] = useState<Phase>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [engineId, setEngineId] = useStoredState<string>("deepocr-engine", "ensemble");
  const [cleanup, setCleanup] = useStoredState<Cleanup>("deepocr-cleanup", "auto");
  const [correctKey, setCorrectKey] = useStoredState<string>("deepocr-correct", "true");
  const correct = correctKey === "true";

  const [running, setRunning] = useState<RunSettings | null>(null);
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [queued, setQueued] = useState(0);
  const [members, setMembers] = useState(0);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [result, setResult] = useState<OcrResponse | null>(null);
  const [doc, setDoc] = useState<DocState | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("lines");
  const [pane, setPane] = useState<"document" | "text">("document");

  const engineList = useMemo(() => engines?.engines ?? [], [engines]);
  const activeEngine = engineList.find((engine) => engine.id === engineId);

  // Fall back to an available engine if the remembered one is not installed.
  useEffect(() => {
    if (!engineList.length || activeEngine?.available) return;
    const fallback = engineList.find((engine) => engine.id === engines?.default && engine.available) ?? sortEngines(engineList).find((engine) => engine.available);
    if (fallback && fallback.id !== engineId) setEngineId(fallback.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engines]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(async (target: File, settings: RunSettings) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(settings);
    setJob(null);
    setQueued(0);
    setError("");
    setPhase("processing");
    window.scrollTo({ top: 0 });
    try {
      const response = await runOcr(
        target,
        { engine: settings.engine, preprocess: CLEANUP_STEPS[settings.cleanup], correct: settings.correct },
        controller.signal,
        {
          onProgress: (snapshot) => {
            setJob(snapshot);
            const count = Object.keys(snapshot.members ?? {}).length;
            if (count) setMembers(count);
          },
          onBusy: setQueued,
        },
      );
      const fresh = initDoc(response);
      const consensus = response.pages.some((page) => page.blocks.some((block) => block.words?.some((word) => word.support !== null)));
      setResult(response);
      setDoc(fresh);
      setPageIndex(0);
      setSelected(null);
      setTab(consensus ? "lines" : "text");
      setPane("document");
      setPhase("results");
      window.scrollTo({ top: 0 });
      onEnginesChanged();
    } catch (caught) {
      if (isAbort(caught)) return;
      setError(caught instanceof Error ? caught.message : "OCR could not process this document.");
      setPhase(result ? "results" : "upload");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [onEnginesChanged, result]);

  const chooseFile = (candidate: File | null) => {
    setError("");
    if (!candidate) {
      setFile(null);
      return;
    }
    const problem = validateFile(candidate);
    if (problem) {
      setFile(null);
      setError(problem);
      return;
    }
    setFile(candidate);
  };

  const start = () => {
    if (!file || !activeEngine?.available) return;
    void run(file, { engine: engineId, cleanup, correct });
  };

  const cancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPhase(result ? "results" : "upload");
  };

  const pickSample = async (sample: SampleDoc) => {
    setLoadingSample(sample.id);
    setError("");
    try {
      const sampleFile = await fetchSampleFile(sample);
      const suggested = engineList.find((engine) => engine.id === sample.suggested_engine && engine.available);
      const engine = suggested?.id ?? engineId;
      const sampleCleanup = sample.preprocess ? cleanupFromSteps(sample.preprocess) : cleanup;
      setFile(sampleFile);
      setEngineId(engine);
      setCleanup(sampleCleanup);
      await run(sampleFile, { engine, cleanup: sampleCleanup, correct });
    } catch (caught) {
      if (!isAbort(caught)) setError(caught instanceof Error ? caught.message : "Could not load the sample.");
    } finally {
      setLoadingSample(null);
    }
  };

  const edited = !!doc && !!result && (doc.markdownOverride !== null || doc.resolved.length > 0 || docMarkdown(doc) !== docMarkdown(initDoc(result)));

  const newDocument = () => {
    if (edited && !window.confirm("Start a new document? Your edits to this one will be lost unless you export them first.")) return;
    abortRef.current?.abort();
    setResult(null);
    setDoc(null);
    setFile(null);
    setError("");
    setPhase("upload");
  };

  const rerun = (engine: string) => {
    if (!file) return;
    if (edited && !window.confirm("Re-run the document? Your edits will be replaced by the new result.")) return;
    setEngineId(engine);
    void run(file, { engine, cleanup: running?.cleanup ?? cleanup, correct: running?.correct ?? correct });
  };

  // Ctrl/Cmd + Enter starts the extraction from the upload screen.
  useEffect(() => {
    if (phase !== "upload") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        start();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const actions: DocActions = useMemo(() => ({
    setText: (p, b, text) => setDoc((state) => (state ? setBlockText(state, p, b, text) : state)),
    replaceWord: (p, b, t, word) => setDoc((state) => (state ? replaceWord(state, p, b, t, word) : state)),
    resolve: (key) => setDoc((state) => (state ? resolve(state, key) : state)),
    setMarkdown: (markdown) => setDoc((state) => (state ? { ...state, markdownOverride: markdown } : state)),
  }), []);

  const select = (p: number, b: number | null) => {
    if (p !== pageIndex) setPageIndex(p);
    setSelected(b);
  };

  /* ---------------------------------------------------------------- processing */
  if (phase === "processing" && file && running) {
    return (
      <div className="page">
        <ProcessingCard
          file={file}
          engine={running.engine}
          steps={CLEANUP_STEPS[running.cleanup].map((step) => ({ deskew: "Straighten", denoise: "Denoise", contrast: "Boost contrast", binarize: "Even out lighting" }[step] ?? step))}
          correct={running.correct}
          job={job}
          queued={queued}
          onCancel={cancel}
        />
      </div>
    );
  }

  /* ---------------------------------------------------------------- results */
  if (phase === "results" && result && doc) {
    const page = result.pages[pageIndex];
    const blocks = doc.pages[pageIndex].blocks;
    const consensus = doc.pages.some((p) => p.blocks.some((block) => block.words?.some((word) => word.support !== null)));
    const engineName = result.engine === "ensemble" ? `consensus of ${members || 5} models` : shortName(result.engine ?? result.model, result.model);
    const pages = result.pages.length;
    return (
      <div className="work">
        <div className="work-head">
          <button type="button" className="btn ghost sm" onClick={newDocument}><ArrowLeft size={15} />New document</button>
          <div className="wh-file">
            <span className="file-ic">{result.filename.toLowerCase().endsWith(".pdf") ? <FileText size={18} /> : <FileImage size={18} />}</span>
            <div>
              <b title={result.filename}>{result.filename}</b>
              <span className="muted">{pages} {pages === 1 ? "page" : "pages"} · {engineName} · auto-correct {result.options?.correct === false ? "off" : "on"}</span>
            </div>
          </div>
          <div className="wh-actions">
            <Menu
              label="Re-run with another engine"
              trigger={({ open, toggle }) => (
                <button type="button" className="btn" aria-expanded={open} aria-haspopup="menu" onClick={toggle} disabled={!file}>
                  <RefreshCw size={15} />Re-run with <b>{shortName(engineId)}</b><ChevronDown size={15} />
                </button>
              )}
            >
              {(close) => (
                <>
                  <span className="label menu-label">Read again with</span>
                  {sortEngines(engineList).map((engine) => (
                    <button key={engine.id} type="button" role="menuitem" disabled={!engine.available} onClick={() => { close(); rerun(engine.id); }}>
                      {engine.id === result.engine ? <Check size={15} /> : <span style={{ width: 15 }} />}
                      {shortName(engine.id, engine.label)}
                      <small>{engine.metrics?.iam ? `${engine.metrics.iam.cer.toFixed(1)}% CER` : ""}</small>
                    </button>
                  ))}
                </>
              )}
            </Menu>
            <ExportMenu result={result} doc={doc} />
          </div>
        </div>

        {error && <div className="alert" role="alert" style={{ marginBottom: 14 }}><AlertCircle size={16} />{error}</div>}
        <SummaryStrip result={result} doc={doc} models={members || 5} />
        <FixBar
          doc={doc}
          onUndo={(p, b, index) => setDoc((state) => {
            if (!state) return state;
            const correction = state.pages[p].blocks[b].corrections?.[index];
            return correction ? undoCorrection(state, p, b, correction) : state;
          })}
          onReview={() => { setTab("review"); setPane("text"); }}
        />

        <div className="mobile-pane-switch">
          <Segmented<"document" | "text">
            label="Show"
            value={pane}
            onChange={setPane}
            options={[{ value: "document", label: "Document" }, { value: "text", label: "Text" }]}
          />
        </div>

        <div className="wgrid" data-pane={pane}>
          <aside className="rail card" aria-label="Pages">
            <span className="label">Pages</span>
            {result.pages.map((p, index) => (
              <button
                key={p.page_number}
                type="button"
                className={`pthumb ${index === pageIndex ? "on" : ""}`}
                aria-label={`Page ${p.page_number}`}
                aria-current={index === pageIndex ? "page" : undefined}
                onClick={() => { setPageIndex(index); setSelected(null); }}
              >
                <img src={`data:image/jpeg;base64,${p.preview_image}`} alt="" />
                <span>{p.page_number}</span>
              </button>
            ))}
            <span className="label" style={{ marginTop: 14 }}>Cleanup</span>
            <CleanupNotes result={result} pageIndex={pageIndex} />
            <span className="rail-note" style={{ marginTop: 10 }}><ShieldCheck size={14} />Processed on this machine</span>
          </aside>
          <DocumentViewer
            page={page}
            blocks={blocks}
            consensus={consensus}
            selected={selected}
            onSelect={(b) => {
              setSelected(b);
              if (b !== null && (tab === "text" || tab === "json")) setTab("lines");
            }}
          />
          <TextPanel
            result={result}
            doc={doc}
            pageIndex={pageIndex}
            tab={tab}
            onTab={setTab}
            selected={selected}
            onSelect={select}
            actions={actions}
          />
        </div>

        <div className="mobile-bar">
          <button type="button" className="btn" onClick={() => setPane(pane === "document" ? "text" : "document")}>
            {pane === "document" ? "Show text" : "Show document"}
          </button>
          <ExportMenu result={result} doc={doc} />
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------- upload */
  const canRun = !!file && !!activeEngine?.available;
  return (
    <div className="page">
      <div className="hero">
        <Pitch engines={engineList} />
        <section className="card extract" aria-label="New extraction">
          <div className="extract-head">
            <h2>New extraction</h2>
            <span className="muted"><ShieldCheck size={15} />Processed locally</span>
          </div>
          <div className="step">
            <div className="step-head"><span className="step-num">1</span>Document</div>
            <Dropzone file={file} onFile={chooseFile} />
          </div>
          <div className="step">
            <div className="step-head"><span className="step-num">2</span>Recognition engine<span className="step-note">Error rates on held-out test sets. Lower is better.</span></div>
            {engineList.length
              ? <EngineList engines={engineList} value={engineId} onChange={setEngineId} />
              : <div className="alert warn"><AlertCircle size={16} />Waiting for the OCR server to list its engines…</div>}
          </div>
          <div className="step">
            <div className="step-head"><span className="step-num">3</span>Options</div>
            <Options cleanup={cleanup} onCleanup={setCleanup} correct={correct} onCorrect={(value) => setCorrectKey(String(value))} />
          </div>
          {error && <div className="alert" role="alert" style={{ marginTop: 18 }}><AlertCircle size={16} />{error}</div>}
          <div className="cta">
            <button type="button" className="btn primary lg block" disabled={!canRun} onClick={start}>
              <ScanText size={18} />
              {!file ? "Add a document to start" : !engineList.length ? "Connecting to the OCR server…" : !activeEngine?.available ? "Choose an available engine" : "Extract text"}
              {canRun && <span className="kbd">Ctrl ↵</span>}
            </button>
          </div>
        </section>
      </div>
      <SampleCards samples={samples} loadingId={loadingSample} disabled={loadingSample !== null} onPick={(sample) => void pickSample(sample)} />
    </div>
  );
}
