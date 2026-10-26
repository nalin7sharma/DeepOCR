import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, FilePlus2, Loader2, RefreshCw, ShieldCheck, Square, Zap } from "lucide-react";
import { fetchSampleFile, recognizeDocument } from "../api";
import { Dropzone, validateFile } from "../components/Dropzone";
import { DocumentViewer } from "../components/DocumentViewer";
import { EnginePicker } from "../components/EnginePicker";
import { OptionsBar } from "../components/OptionsBar";
import { PipelineViz } from "../components/PipelineViz";
import { ResultPanel } from "../components/ResultPanel";
import { SampleGallery } from "../components/SampleGallery";
import { StatsRibbon } from "../components/StatsRibbon";
import { useStoredState } from "../hooks";
import type { EnginesResponse, OcrResponse, SampleDoc, StructuredBlock } from "../types";

interface Props {
  engines: EnginesResponse | null;
  samples: SampleDoc[];
  onEnginesChanged: () => void;
}

function ScanPreview({ file }: { file: File | null }) {
  const url = useMemo(() => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);
  return (
    <div className="scan-preview" aria-hidden="true">
      {url ? <img src={url} alt="" /> : <div className="scan-placeholder">{file?.name.toLowerCase().endsWith(".pdf") ? "PDF" : "DOC"}</div>}
      <span className="laser" />
      <span className="scan-grid" />
    </div>
  );
}

const HEADLINE_WORDS = ["handwritten notes", "scanned forms", "invoices & tables", "noisy, skewed scans"];

export function Studio({ engines, samples, onEnginesChanged }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<OcrResponse | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState(0);
  const [pageIndex, setPageIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [engineId, setEngineId] = useStoredState<string>("deepocr-engine", "paddle");
  const [preprocessKey, setPreprocessKey] = useStoredState<string>("deepocr-preprocess", "deskew");
  const [correctKey, setCorrectKey] = useStoredState<string>("deepocr-correct", "true");
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [wordIndex, setWordIndex] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const preprocess = preprocessKey ? preprocessKey.split(",").filter(Boolean) : [];
  const correct = correctKey === "true";
  const engineList = engines?.engines ?? [];
  const activeEngine = engineList.find((engine) => engine.id === engineId);

  useEffect(() => {
    const timer = window.setInterval(() => setWordIndex((i) => (i + 1) % HEADLINE_WORDS.length), 2400);
    return () => window.clearInterval(timer);
  }, []);

  // Fall back to an available engine if the remembered one is unavailable.
  useEffect(() => {
    if (!engineList.length) return;
    if (!activeEngine?.available) {
      const fallback = engineList.find((engine) => engine.id === engines?.default && engine.available) ?? engineList.find((engine) => engine.available);
      if (fallback && fallback.id !== engineId) setEngineId(fallback.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engines]);

  function reset() {
    abortRef.current?.abort();
    setResult(null);
    setDraft("");
    setError("");
    setPageIndex(0);
    setSelected(null);
  }

  function chooseFile(candidate: File | null) {
    reset();
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
  }

  async function run(target: File | null = file, overrides?: { engine?: string; preprocess?: string[]; correct?: boolean }) {
    if (!target || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError("");
    setQueued(0);
    setSelected(null);
    try {
      const response = await recognizeDocument(
        target,
        { engine: overrides?.engine ?? engineId, preprocess: overrides?.preprocess ?? preprocess, correct: overrides?.correct ?? correct },
        controller.signal,
        setQueued,
      );
      setResult(response);
      setDraft(response.markdown);
      setPageIndex(0);
      onEnginesChanged();
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "OCR could not process this document.");
    } finally {
      setBusy(false);
      setQueued(0);
    }
  }

  async function pickSample(sample: SampleDoc) {
    if (busy) return;
    setLoadingSample(sample.id);
    try {
      const sampleFile = await fetchSampleFile(sample);
      reset();
      setFile(sampleFile);
      const engine = sample.suggested_engine && engineList.find((e) => e.id === sample.suggested_engine && e.available)
        ? sample.suggested_engine
        : engineId;
      if (engine !== engineId) setEngineId(engine);
      const steps = sample.preprocess ?? preprocess;
      if (sample.preprocess) setPreprocessKey(sample.preprocess.join(","));
      await run(sampleFile, { engine, preprocess: steps });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load the sample.");
    } finally {
      setLoadingSample(null);
    }
  }

  const selectBlock = (index: number | null, _block?: StructuredBlock) => setSelected(index);

  const controls = (
    <div className="control-deck">
      <div className="deck-row">
        <span className="section-label"><Zap size={14} /> Recognition engine</span>
        {activeEngine && <span className="deck-hint">{activeEngine.description}</span>}
      </div>
      <EnginePicker engines={engineList} value={engineId} onChange={setEngineId} />
      <OptionsBar
        preprocess={preprocess}
        onPreprocess={(steps) => setPreprocessKey(steps.join(","))}
        correct={correct}
        onCorrect={(value) => setCorrectKey(String(value))}
      />
    </div>
  );

  return (
    <div className="studio">
      {!result && !busy && (
        <section className="hero">
          <div className="hero-copy">
            <motion.p className="eyebrow" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <span className="pulse-dot" /> Deep learning OCR · CNN-BiLSTM · CRNN · Transformer
            </motion.p>
            <h1>
              Turn{" "}
              <span className="rotator">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={wordIndex}
                    className="gradient-text"
                    initial={{ y: "60%", opacity: 0, filter: "blur(6px)" }}
                    animate={{ y: "0%", opacity: 1, filter: "blur(0px)" }}
                    exit={{ y: "-60%", opacity: 0, filter: "blur(6px)" }}
                    transition={{ duration: 0.45 }}
                  >
                    {HEADLINE_WORDS[wordIndex]}
                  </motion.span>
                </AnimatePresence>
              </span>
              <br />into clean, structured text.
            </h1>
            <p className="hero-sub">
              Preprocessing, text detection, layout analysis, deep-learning recognition and error correction, in one
              reviewable pipeline. Every uncertain word is flagged and every automatic fix is shown.
            </p>
            <ul className="hero-points">
              <li><ShieldCheck size={16} /> In-memory processing. Nothing is stored.</li>
              <li><Zap size={16} /> Four engines, including models trained for this project.</li>
            </ul>
          </div>

          <div className="hero-panel glass">
            <Dropzone file={file} onFile={chooseFile} />
            {controls}
            <div className="run-row">
              {busy ? (
                <button type="button" className="ghost-button" onClick={() => abortRef.current?.abort()}><Square size={14} /> Cancel</button>
              ) : <span className="run-note">{file ? "Ready when you are." : "Pick a file or a demo below."}</span>}
              <button type="button" className="primary-button big" disabled={!file || busy} onClick={() => run()}>
                {busy ? <><Loader2 className="spin" size={18} /> Reading…</> : <>Extract text <ArrowRight size={18} /></>}
              </button>
            </div>
            {error && <div className="error-banner" role="alert">{error}</div>}
          </div>
        </section>
      )}

      {!result && !busy && <SampleGallery samples={samples} loadingId={loadingSample} onPick={pickSample} />}

      {busy && (
        <motion.section className="processing glass" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}>
          <div className="processing-grid">
            <ScanPreview file={file} />
            <div className="processing-side">
              <p className="eyebrow"><span className="pulse-dot" /> Reading {file?.name}</p>
              <h2 className="processing-title">{activeEngine?.label ?? "OCR"} is working through your document…</h2>
              <PipelineViz running engineLabel={activeEngine?.label} />
              {queued > 0 && <p className="queue-note">The server is finishing another document. Retrying automatically (attempt {queued})…</p>}
              <button type="button" className="ghost-button" onClick={() => abortRef.current?.abort()}><Square size={14} /> Cancel</button>
            </div>
          </div>
        </motion.section>
      )}

      {result && (
        <>
          <section className="result-head">
            <div className="file-identity">
              <span className="file-badge">{result.filename.toLowerCase().endsWith(".pdf") ? "PDF" : "IMG"}</span>
              <div>
                <strong title={result.filename}>{result.filename}</strong>
                <span>{result.pages.length} {result.pages.length === 1 ? "page" : "pages"} · {result.model}{result.options?.preprocess?.length ? ` · ${result.options.preprocess.join(" + ")}` : ""}{result.options?.correct ? " · auto-correct" : ""}</span>
              </div>
            </div>
            <div className="result-actions">
              <select
                className="select"
                aria-label="Re-run with engine"
                value={engineId}
                onChange={(event) => setEngineId(event.target.value)}
              >
                {engineList.map((engine) => (
                  <option key={engine.id} value={engine.id} disabled={!engine.available}>{engine.label}</option>
                ))}
              </select>
              <button type="button" className="ghost-button" disabled={busy} onClick={() => run()}><RefreshCw size={15} /> Re-run</button>
              <button type="button" className="ghost-button" onClick={() => { chooseFile(null); }}><FilePlus2 size={15} /> New document</button>
            </div>
          </section>

          {!busy && <PipelineViz running={false} timings={result.timings} engineLabel={result.model} />}
          <StatsRibbon result={result} />

          <section className="workbench">
            <div className="glass pane">
              <DocumentViewer
                pages={result.pages}
                pageIndex={pageIndex}
                onPageChange={(index) => { setPageIndex(index); setSelected(null); }}
                selected={selected}
                onSelect={selectBlock}
              />
            </div>
            <div className="glass pane">
              <ResultPanel
                result={result}
                draft={draft}
                onDraft={setDraft}
                pageIndex={pageIndex}
                selected={selected}
                onSelect={selectBlock}
                onError={setError}
              />
            </div>
          </section>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </>
      )}
    </div>
  );
}
