import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, FileText, Loader2, Play, Square, Timer, Trophy } from "lucide-react";
import { fetchSampleFile, isAbort, runOcr } from "../api";
import { Dropzone, useFilePreview, validateFile } from "../components/Dropzone";
import { EngineChips } from "../components/EngineList";
import { CLEANUP_STEPS, cleanupFromSteps } from "../components/Options";
import { isDisputed } from "../lib/document";
import { colorFor, shortName, sortEngines } from "../lib/engines";
import { cer, wer, wordDiff, type WordToken } from "../lib/text";
import type { EnginesResponse, JobSnapshot, OcrResponse, SampleDoc, StructuredBlock, WordVote } from "../types";

interface Run {
  engine: string;
  status: "waiting" | "running" | "done" | "error";
  job?: JobSnapshot | null;
  result?: OcrResponse;
  error?: string;
}

const DEFAULT_SELECTION = ["ensemble", "trocr_ft", "crnn", "cnn_bilstm", "paddle"];

const blockLine = (block: StructuredBlock) =>
  block.kind === "illegible" ? block.raw_text ?? ""
    : block.kind === "table" && block.cells ? block.cells.map((row) => row.join(" ")).join(" ")
      : block.text;

const resultLines = (result: OcrResponse) =>
  result.pages.flatMap((page) => page.blocks.map(blockLine)).map((line) => line.trim()).filter(Boolean);

const seconds = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

function Tokens({ tokens }: { tokens: WordToken[] }) {
  return (
    <p>
      {tokens.map((token, index) => (
        <span key={index}>
          {token.br && <br />}
          {index > 0 && !token.br && " "}
          {token.type === "ok" && token.text}
          {token.type === "wrong" && <span className="wrong" title={token.expected ? `Expected: ${token.expected}` : "Extra word, not in the ground truth"}>{token.text}</span>}
          {token.type === "miss" && <span className="miss" title={`Missing: ${token.expected}`}>‸</span>}
        </span>
      ))}
    </p>
  );
}

/** Without ground truth: the consensus column underlines the words the models disputed. */
function PlainLines({ result }: { result: OcrResponse }) {
  return (
    <p>
      {result.pages.flatMap((page) => page.blocks).map((block, b) => {
        const tokens = block.text.split(/\s+/).filter(Boolean);
        const votes: (WordVote | null)[] = block.words && block.words.length === tokens.length ? block.words : tokens.map(() => null);
        const line = blockLine(block).trim();
        if (!line) return null;
        return (
          <span key={b}>
            {b > 0 && <br />}
            {block.kind === "illegible" || block.kind === "table" ? line : tokens.map((token, t) => {
              const vote = votes[t];
              const alternatives = vote ? Object.keys(vote.alternatives).slice(0, 3).join(", ") : "";
              return (
                <span key={t}>
                  {t > 0 && " "}
                  {vote && isDisputed(vote) ? <span className="dis" style={{ cursor: "help" }} title={`Models disagreed${alternatives ? `; also read as: ${alternatives}` : ""}`}>{token}</span> : token}
                </span>
              );
            })}
          </span>
        );
      })}
    </p>
  );
}

function RunningBody({ run }: { run: Run }) {
  const job = run.job;
  if (run.status === "waiting") {
    return <div className="col-loading"><Timer size={20} />Waiting for its turn<span className="hint">Engines run one after another so each gets the full CPU and a fair time.</span></div>;
  }
  const progress = job && job.lines_total ? job.lines_done / job.lines_total : null;
  const members = Object.values(job?.members ?? {});
  const finished = members.filter((m) => m.total > 0 && m.done >= m.total).length;
  return (
    <div className="col-loading">
      <Loader2 size={20} className="spin" />
      <b style={{ color: "var(--text)" }}>{job?.stage_label || "Starting…"}</b>
      {members.length > 0 && <span>{finished} of {members.length} models finished</span>}
      {job && job.lines_total > 0 && <span>{Math.min(job.lines_done, job.lines_total)} of {job.lines_total} lines</span>}
      <span className={`bar ${progress === null ? "indeterminate" : ""}`} style={{ width: "70%" }}><i style={progress === null ? undefined : { width: `${Math.max(4, progress * 100)}%` }} /></span>
    </div>
  );
}

export function Compare({ engines, samples }: { engines: EnginesResponse | null; samples: SampleDoc[] }) {
  const engineList = useMemo(() => engines?.engines ?? [], [engines]);
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<SampleDoc | null>(null);
  const [selection, setSelection] = useState<string[]>(DEFAULT_SELECTION);
  const [reference, setReference] = useState("");
  const [showGt, setShowGt] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const preview = useFilePreview(file);

  useEffect(() => () => abortRef.current?.abort(), []);

  const running = runs.some((run) => run.status === "running" || run.status === "waiting");
  const chosen = sortEngines(engineList).map((engine) => engine.id).filter((id) => selection.includes(id) && engineList.find((engine) => engine.id === id)?.available);
  const hasReference = reference.trim().length > 0;
  const referenceLines = reference.split("\n").filter((line) => line.trim()).length;

  const update = (index: number, change: Partial<Run>) =>
    setRuns((current) => current.map((run, i) => (i === index ? { ...run, ...change } : run)));

  async function start(target: File | null = file, source: SampleDoc | null = sample) {
    if (!target || !chosen.length) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError("");
    const queue: Run[] = chosen.map((engine) => ({ engine, status: "waiting" }));
    setRuns(queue);
    // Auto-correct stays off so each model's own reading is compared.
    const preprocess = CLEANUP_STEPS[source?.preprocess ? cleanupFromSteps(source.preprocess) : "auto"];
    for (let index = 0; index < queue.length; index += 1) {
      if (controller.signal.aborted) return;
      update(index, { status: "running", job: null });
      try {
        const result = await runOcr(target, { engine: queue[index].engine, preprocess, correct: false }, controller.signal, {
          onProgress: (job) => update(index, { job }),
        });
        update(index, { status: "done", result });
      } catch (caught) {
        if (isAbort(caught)) return;
        update(index, { status: "error", error: caught instanceof Error ? caught.message : "Failed" });
      }
    }
  }

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRuns((current) => current.map((run) => (run.status === "waiting" || run.status === "running" ? { ...run, status: "error", error: "Cancelled" } : run)));
  };

  const chooseFile = (candidate: File | null) => {
    stop();
    setRuns([]);
    setSample(null);
    setError("");
    if (!candidate) {
      setFile(null);
      return;
    }
    const problem = validateFile(candidate);
    if (problem) {
      setError(problem);
      return;
    }
    setFile(candidate);
    setReference("");
    setShowGt(true);
  };

  const pickSample = async (doc: SampleDoc) => {
    setLoadingSample(doc.id);
    setError("");
    try {
      const sampleFile = await fetchSampleFile(doc);
      stop();
      setFile(sampleFile);
      setSample(doc);
      setReference(doc.ground_truth ?? "");
      setShowGt(!doc.ground_truth);
      void start(sampleFile, doc);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load the sample.");
    } finally {
      setLoadingSample(null);
    }
  };

  const scored = useMemo(() => runs.map((run) => {
    if (!run.result) return { ...run, lines: [] as string[], cer: null, wer: null, time: null, tokens: null };
    const lines = resultLines(run.result);
    const text = lines.join(" ");
    return {
      ...run,
      lines,
      cer: hasReference ? cer(text, reference) : null,
      wer: hasReference ? wer(text, reference) : null,
      time: run.result.timings?.total ?? run.result.inference_ms,
      tokens: hasReference ? wordDiff(lines, reference) : null,
    };
  }), [runs, reference, hasReference]);

  const done = scored.filter((run) => run.status === "done");
  const best = hasReference && done.length > 1 ? done.reduce((a, b) => ((a.cer ?? 1) <= (b.cer ?? 1) ? a : b)).engine : null;
  const fastest = done.length > 1 ? done.reduce((a, b) => ((a.time ?? Infinity) <= (b.time ?? Infinity) ? a : b)).engine : null;
  const maxCer = Math.max(0.01, ...done.map((run) => run.cer ?? 0));
  const firstResult = done[0]?.result;
  const lineCount = firstResult ? firstResult.stats?.lines ?? resultLines(firstResult).length : null;
  const columns = scored.length <= 5 ? Math.max(1, scored.length) : 3;

  return (
    <div className="page">
      <div className="ph">
        <h1>Compare engines</h1>
        <p>Run several models on the same document, side by side with their consensus vote. Add the correct text to score each one word by word.</p>
      </div>

      {!file ? (
        <section className="card cmp-start" aria-label="Choose a document">
          <div>
            <div className="step-head"><span className="step-num">1</span>Your document</div>
            <Dropzone file={null} onFile={chooseFile} />
          </div>
          <div>
            <div className="step-head"><span className="step-num">or</span>A sample with known text</div>
            <div className="cmp-samples">
              {samples.map((doc) => (
                <button key={doc.id} type="button" className="cmp-sample" disabled={loadingSample !== null} onClick={() => void pickSample(doc)}>
                  <img src={`/api/v1/samples/${encodeURIComponent(doc.id)}/thumb`} alt="" loading="lazy" />
                  <span>
                    <b>{doc.title}</b>
                    <span className="muted">{doc.ground_truth ? "Ground truth included, scored automatically" : doc.subtitle}</span>
                  </span>
                  {loadingSample === doc.id && <Loader2 size={16} className="spin" style={{ marginLeft: "auto" }} />}
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : (
        <>
          <section className="card setup" aria-label="Comparison setup">
            <div className="doc-chip">
              {preview ? <img src={preview} alt="" /> : sample ? <img src={`/api/v1/samples/${encodeURIComponent(sample.id)}/thumb`} alt="" /> : <span className="file-ic"><FileText size={18} /></span>}
              <div>
                <b title={file.name}>{file.name}</b>
                <span className="muted">{firstResult ? `${firstResult.pages.length} ${firstResult.pages.length === 1 ? "page" : "pages"} · ${lineCount} lines` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</span>
              </div>
              <button type="button" className="btn sm ghost" onClick={() => chooseFile(null)}>Change</button>
            </div>
            <span className="vsep" />
            <EngineChips engines={engineList} value={selection} onChange={setSelection} />
            <span className="vsep" />
            <div className="gt">
              <span className="label">Ground truth</span>
              <button type="button" className={`badge ${hasReference ? "success" : "outline"}`} onClick={() => setShowGt(!showGt)} aria-expanded={showGt}>
                {hasReference ? `${referenceLines} ${referenceLines === 1 ? "line" : "lines"} ${sample?.ground_truth === reference ? "included" : "pasted"}` : "Add text"}
              </button>
            </div>
            {running
              ? <button type="button" className="btn run" onClick={stop}><Square size={14} />Stop</button>
              : <button type="button" className="btn primary run" disabled={!chosen.length} onClick={() => void start()}><Play size={15} />{runs.length ? "Run again" : `Run ${chosen.length} ${chosen.length === 1 ? "engine" : "engines"}`}</button>}
          </section>

          {showGt && (
            <section className="card gt-panel" aria-label="Ground truth">
              <div className="step-head">The correct text<span className="step-note">Optional. Paste what the document really says to get CER, WER and a word-by-word diff.</span></div>
              <textarea value={reference} onChange={(event) => setReference(event.target.value)} placeholder="Paste or type the exact text of the document…" aria-label="Ground-truth text" />
            </section>
          )}
        </>
      )}

      {error && <div className="alert" role="alert" style={{ marginTop: 14 }}><AlertCircle size={16} />{error}</div>}

      {scored.length > 0 && (
        <>
          <div className="legend-row">
            {hasReference ? (
              <>
                <span className="muted">Compared word by word with the ground truth:</span>
                <span className="wrong">misread word</span><span className="muted">hover to see the correct text</span>
                <span className="miss">‸</span><span className="muted">missing word</span>
              </>
            ) : (
              <>
                <span className="muted">No ground truth yet, so there are no scores.</span>
                <span className="dis">underlined</span><span className="muted">words were disputed in the consensus vote</span>
              </>
            )}
          </div>
          <div className="cols" style={{ ["--n" as string]: columns }}>
            {scored.map((run) => {
              const color = colorFor(run.engine);
              const consensusWords = run.result?.pages.flatMap((page) => page.blocks.flatMap((block) => block.words ?? [])).filter((w) => w.support !== null) ?? [];
              const agreement = consensusWords.length ? consensusWords.filter((w) => !isDisputed(w)).length / consensusWords.length : null;
              return (
                <article key={run.engine} className={`col card ${best === run.engine ? "best" : ""}`}>
                  <div className="col-head" style={{ ["--c" as string]: color }}>
                    <div className="col-name"><i /><b>{shortName(run.engine)}</b></div>
                    <div className="col-badges">
                      {best === run.engine && <span className="badge success"><Trophy size={12} />Most accurate</span>}
                      {fastest === run.engine && <span className="badge accent"><Timer size={12} />Fastest</span>}
                    </div>
                    <div className="col-metrics">
                      {hasReference ? (
                        <>
                          <div><span className="label">CER</span><b className="num">{run.cer !== null ? `${(run.cer * 100).toFixed(1)}%` : "—"}</b></div>
                          <div><span className="label">WER</span><b className="num">{run.wer !== null ? `${(run.wer * 100).toFixed(1)}%` : "—"}</b></div>
                        </>
                      ) : (
                        <>
                          <div>
                            <span className="label">{agreement !== null ? "Agree" : "Conf."}</span>
                            <b className="num">{run.result ? `${Math.round((agreement ?? run.result.stats?.mean_confidence ?? 0) * 100)}%` : "—"}</b>
                          </div>
                          <div><span className="label">Words</span><b className="num">{run.result?.stats?.words ?? "—"}</b></div>
                        </>
                      )}
                      <div><span className="label">Time</span><b className="num">{run.time !== null ? seconds(run.time) : "—"}</b></div>
                    </div>
                    {hasReference && <span className="cerbar" title="Character error rate"><i style={{ width: `${run.cer !== null ? Math.max(2, (run.cer / maxCer) * 100) : 0}%`, background: color }} /></span>}
                  </div>
                  <div className="col-body">
                    {run.status === "error" && <div className="alert"><AlertCircle size={16} />{run.error}</div>}
                    {(run.status === "waiting" || run.status === "running") && <RunningBody run={run} />}
                    {run.status === "done" && run.result && (
                      run.tokens ? <Tokens tokens={run.tokens} /> : run.lines.length ? <PlainLines result={run.result} /> : <p className="muted">No text recognised.</p>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
