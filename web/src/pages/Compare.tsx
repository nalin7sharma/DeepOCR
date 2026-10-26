import { useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Crown, Gauge as GaugeIcon, Loader2, Play, Square, Timer, Trophy } from "lucide-react";
import { fetchSampleFile, recognizeDocument } from "../api";
import { Dropzone, validateFile } from "../components/Dropzone";
import { EnginePicker } from "../components/EnginePicker";
import { OptionsBar } from "../components/OptionsBar";
import { colorFor } from "../components/charts";
import type { EnginesResponse, OcrResponse, SampleDoc } from "../types";
import { cer, charDiff, ms, plainText, wer } from "../lib/text";

interface Run {
  engine: string;
  status: "waiting" | "running" | "done" | "error";
  result?: OcrResponse;
  error?: string;
}

function DiffView({ prediction, reference }: { prediction: string; reference: string }) {
  const lines = prediction.split("\n");
  const refLines = reference.split("\n");
  return (
    <div className="diff-view">
      {lines.map((line, index) => (
        <div key={index} className="diff-line">
          {charDiff(line, refLines[index] ?? "").map((op, i) => (
            op.type === "equal" ? <span key={i}>{op.text}</span>
              : op.type === "insert" ? <span key={i} className="diff-ins" title="Not in reference">{op.text}</span>
              : <span key={i} className="diff-del" title="Missing from output">{op.text}</span>
          ))}
        </div>
      ))}
    </div>
  );
}

export function Compare({ engines, samples }: { engines: EnginesResponse | null; samples: SampleDoc[] }) {
  const engineList = engines?.engines ?? [];
  const [file, setFile] = useState<File | null>(null);
  const [selection, setSelection] = useState<string[]>(() => ["paddle", "cnn_bilstm", "crnn", "trocr_ft"]);
  const [preprocess, setPreprocess] = useState<string[]>(["deskew"]);
  const [correct, setCorrect] = useState(false);
  const [reference, setReference] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const running = runs.some((run) => run.status === "running" || run.status === "waiting");
  const chosen = selection.filter((id) => engineList.find((engine) => engine.id === id)?.available);

  async function start(target: File | null = file) {
    if (!target || !chosen.length) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setError("");
    const queue: Run[] = chosen.map((engine) => ({ engine, status: "waiting" }));
    setRuns(queue);
    for (let index = 0; index < queue.length; index += 1) {
      if (controller.signal.aborted) break;
      setRuns((current) => current.map((run, i) => (i === index ? { ...run, status: "running" } : run)));
      try {
        const result = await recognizeDocument(target, { engine: queue[index].engine, preprocess, correct }, controller.signal);
        setRuns((current) => current.map((run, i) => (i === index ? { ...run, status: "done", result } : run)));
      } catch (caught) {
        if (caught instanceof DOMException && caught.name === "AbortError") break;
        const message = caught instanceof Error ? caught.message : "Failed";
        setRuns((current) => current.map((run, i) => (i === index ? { ...run, status: "error", error: message } : run)));
      }
    }
    setRuns((current) => current.map((run) => (run.status === "waiting" || run.status === "running" ? { ...run, status: "error", error: "Cancelled" } : run)));
  }

  const scored = useMemo(() => runs.map((run) => {
    const text = run.result ? plainText(run.result.markdown) : "";
    return {
      ...run,
      text,
      cer: run.result && reference.trim() ? cer(text, reference) : null,
      wer: run.result && reference.trim() ? wer(text, reference) : null,
    };
  }), [runs, reference]);
  const finished = scored.filter((run) => run.status === "done");
  const bestCer = finished.length && reference.trim() ? Math.min(...finished.map((run) => run.cer ?? Infinity)) : null;
  const fastest = finished.length ? Math.min(...finished.map((run) => run.result?.timings?.total ?? Infinity)) : null;

  return (
    <div className="compare">
      <header className="page-head">
        <p className="eyebrow"><span className="pulse-dot" /> Engine battle</p>
        <h1>Same document. <span className="gradient-text">Different brains.</span></h1>
        <p className="hero-sub">Run the pretrained reference, our trained CNN-BiLSTM and CRNN, and the Transformer side by side. Paste the true text to score each one with CER and WER.</p>
      </header>

      <section className="compare-setup">
        <div className="glass pane setup-left">
          <Dropzone
            file={file}
            compact
            onFile={(candidate) => {
              setRuns([]);
              if (candidate) {
                const problem = validateFile(candidate);
                setError(problem);
                setFile(problem ? null : candidate);
              } else setFile(null);
            }}
          />
          {samples.length > 0 && (
            <div className="sample-chips">
              {samples.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  className="chip"
                  disabled={running}
                  onClick={async () => {
                    try {
                      const sampleFile = await fetchSampleFile(sample);
                      setFile(sampleFile);
                      setRuns([]);
                      if (sample.preprocess) setPreprocess(sample.preprocess);
                    } catch (caught) {
                      setError(caught instanceof Error ? caught.message : "Could not load sample");
                    }
                  }}
                >
                  {sample.title}
                </button>
              ))}
            </div>
          )}
          <label className="field">
            <span>Ground truth (optional). Paste the exact text to score each engine.</span>
            <textarea className="editor small" value={reference} onChange={(event) => setReference(event.target.value)} placeholder="Type or paste the text that is actually on the page…" />
          </label>
        </div>
        <div className="glass pane setup-right">
          <span className="section-label">Contenders</span>
          <EnginePicker engines={engineList} value={selection} onChange={setSelection} multiple />
          <OptionsBar preprocess={preprocess} onPreprocess={setPreprocess} correct={correct} onCorrect={setCorrect} />
          <div className="run-row">
            {running
              ? <button type="button" className="ghost-button" onClick={() => abortRef.current?.abort()}><Square size={14} /> Stop</button>
              : <span className="run-note">{chosen.length} engine{chosen.length === 1 ? "" : "s"} selected</span>}
            <button type="button" className="primary-button big" disabled={!file || !chosen.length || running} onClick={() => start()}>
              {running ? <><Loader2 className="spin" size={18} /> Running…</> : <><Play size={17} /> Start battle</>}
            </button>
          </div>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </div>
      </section>

      {scored.length > 0 && (
        <section className="battle-grid" style={{ ["--cols" as string]: Math.min(scored.length, 4) }}>
          {scored.map((run, index) => {
            const engine = engineList.find((item) => item.id === run.engine);
            const total = run.result?.timings?.total;
            const isBest = bestCer !== null && run.cer === bestCer;
            const isFastest = fastest !== null && total === fastest && finished.length > 1;
            return (
              <motion.article
                key={run.engine}
                className={`glass battle-card ${run.status} ${isBest ? "winner" : ""}`}
                style={{ ["--accent" as string]: colorFor(run.engine, index) }}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.08 }}
              >
                <header>
                  <span className="battle-name"><i />{engine?.label ?? run.engine}</span>
                  {isBest && <span className="tag win"><Crown size={12} /> lowest CER</span>}
                  {isFastest && <span className="tag"><Trophy size={12} /> fastest</span>}
                </header>
                {run.status === "running" && <div className="battle-loading"><Loader2 className="spin" size={22} /><span>Reading…</span><span className="scan-bar" /></div>}
                {run.status === "waiting" && <div className="battle-loading muted">In queue</div>}
                {run.status === "error" && <div className="error-banner">{run.error}</div>}
                {run.status === "done" && run.result && (
                  <>
                    <div className="battle-stats">
                      <span><Timer size={13} /> {ms(total)}</span>
                      <span><GaugeIcon size={13} /> {Math.round((run.result.stats?.mean_confidence ?? 0) * 100)}% conf</span>
                      {run.cer !== null && <span className="metric">CER {(run.cer * 100).toFixed(1)}%</span>}
                      {run.wer !== null && <span className="metric">WER {(run.wer * 100).toFixed(1)}%</span>}
                    </div>
                    {reference.trim()
                      ? <DiffView prediction={run.text} reference={reference.trim()} />
                      : <pre className="battle-text">{run.text || "No text recognised."}</pre>}
                  </>
                )}
              </motion.article>
            );
          })}
        </section>
      )}
      {reference.trim() && finished.length > 0 && (
        <p className="legend-note"><span className="diff-ins">extra</span> characters the engine added · <span className="diff-del">missing</span> characters it dropped (compared line by line with your ground truth)</p>
      )}
    </div>
  );
}
