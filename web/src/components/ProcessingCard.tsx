import { useEffect, useState } from "react";
import { Check, Clock, FileImage, FileText, Loader2, X } from "lucide-react";
import type { JobSnapshot } from "../types";
import { shortName } from "../lib/engines";
import { useFilePreview } from "./Dropzone";

interface Props {
  file: File;
  engine: string;
  steps: string[];
  correct: boolean;
  job: JobSnapshot | null;
  queued: number;
  onCancel: () => void;
}

const ORDER: Record<string, number> = {
  queued: 0, decode: 0, preprocess: 1, detect: 2, loading: 3, recognize: 3, vote: 4, layout: 5, postprocess: 6, done: 7,
};

const fmt = (ms?: number) => (ms === undefined ? "" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.max(1, Math.round(ms))} ms`);

export function ProcessingCard({ file, engine, steps, correct, job, queued, onCancel }: Props) {
  const preview = useFilePreview(file);
  const [now, setNow] = useState(() => performance.now());
  const [start] = useState(() => performance.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 200);
    return () => window.clearInterval(timer);
  }, []);

  const ensemble = engine === "ensemble";
  const current = job ? ORDER[job.stage] ?? 0 : -1;
  const members = Object.entries(job?.members ?? {});
  const lines = job && job.lines_total > 0 ? `${Math.min(job.lines_done, job.lines_total)} of ${job.lines_total} lines` : "";
  const times = job?.stage_times ?? {};

  const recognizeDetail = (() => {
    if (job?.stage === "loading") return "Loading the model (first use only)…";
    if (ensemble) {
      if (!members.length) return "Every model reads each line";
      const finished = members.filter(([, m]) => m.total > 0 && m.done >= m.total).length;
      return `${finished} of ${members.length} models have read every line`;
    }
    if (engine === "paddle") return "Detecting and reading lines";
    return lines ? `${shortName(engine)} has read ${lines}` : `${shortName(engine)} reads each line`;
  })();

  const rows: { key: string; title: string; detail: string; index: number; time?: number; show?: boolean }[] = [
    { key: "decode", title: job ? "Read file" : "Upload file", detail: job?.pages ? `${job.pages} ${job.pages === 1 ? "page" : "pages"} decoded` : queued ? `Waiting for the server (attempt ${queued})…` : "Sending the document", index: 0, time: times.decode },
    { key: "preprocess", title: "Clean up image", detail: steps.length ? steps.join(" · ") : "Skipped (cleanup off)", index: 1, time: times.preprocess },
    { key: "detect", title: "Find text lines", detail: job && job.lines_total ? `${job.lines_total} lines detected` : "Text-line detection (DBNet)", index: 2, time: times.detect },
    { key: "recognize", title: ensemble ? `Read with ${members.length || 5} models` : "Recognise text", detail: recognizeDetail, index: 3, time: (times.recognize ?? 0) + (times.loading ?? 0) || undefined },
    { key: "vote", title: "Vote on each word", detail: "Align the readings and keep the words the models agree on", index: 4, time: times.vote, show: ensemble },
    { key: "layout", title: "Rebuild layout", detail: "Reading order, headings, tables, checkboxes", index: 5, time: times.layout },
    { key: "postprocess", title: "Correct errors", detail: correct ? "Spelling and character fixes" : "Skipped (auto-correct off)", index: 6, time: times.postprocess },
  ];

  const elapsed = job ? Math.max(job.elapsed_ms, now - start) : now - start;
  const progress = job && job.lines_total ? (job.lines_done / job.lines_total) * 100 : null;

  return (
    <section className="card proc" aria-live="polite">
      <div className="proc-doc">
        <div className="paper">
          {preview ? <img src={preview} alt={`Document being read: ${file.name}`} /> : <div className="placeholder">PDF</div>}
          <span className="beam" aria-hidden="true" />
        </div>
      </div>
      <div className="proc-side">
        <div className="proc-file">
          <span className="file-ic">{preview ? <FileImage size={18} /> : <FileText size={18} />}</span>
          <div>
            <b title={file.name}>{file.name}</b>
            <span className="muted">{job?.pages && job.pages > 1 ? `Page ${job.page || 1} of ${job.pages}` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</span>
          </div>
          <span className="badge accent">{shortName(engine)}</span>
        </div>
        <h2>Reading your document…</h2>
        <ol className="psteps">
          {rows.filter((row) => row.show !== false).map((row) => {
            const state = current > row.index ? "done" : current === row.index ? "active" : row.index === 0 && current === -1 ? "active" : "wait";
            return (
              <li key={row.key} className={`ps ${state}`}>
                <span className="ps-dot">{state === "done" && <Check size={13} strokeWidth={3} />}</span>
                <span className="ps-main">
                  <b>{row.title}</b>
                  <span>{row.detail}</span>
                  {state === "active" && row.key === "recognize" && ensemble && members.length > 0 && (
                    <span className="members">
                      {members.map(([id, m]) => {
                        const finished = m.total > 0 && m.done >= m.total;
                        return (
                          <span key={id} className={`mem ${finished ? "done" : "active"}`}>
                            {finished ? <Check size={12} /> : <Loader2 size={12} className="spin" />}
                            {shortName(id)}{!finished && m.total > 0 && <em>{m.done}/{m.total}</em>}
                          </span>
                        );
                      })}
                    </span>
                  )}
                  {state === "active" && row.key === "recognize" && (
                    <span className={`bar ${progress === null ? "indeterminate" : ""}`}><i style={progress === null ? undefined : { width: `${Math.max(4, progress)}%` }} /></span>
                  )}
                </span>
                <span className="ps-time num">{state === "done" ? fmt(row.time) : ""}</span>
              </li>
            );
          })}
        </ol>
        <div className="proc-foot">
          <span className="muted"><Clock size={14} /> {(elapsed / 1000).toFixed(1)} s elapsed{ensemble ? " · Consensus runs every model, so it is the slowest and most accurate option" : ""}</span>
          <button type="button" className="btn sm" onClick={onCancel}><X size={14} />Cancel</button>
        </div>
      </div>
    </section>
  );
}
