import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, FileInput, LayoutPanelTop, ScanSearch, SpellCheck2, Wand2, Type } from "lucide-react";
import { ms } from "../lib/text";

export const STAGES = [
  { id: "decode", label: "Decode", detail: "PDF / image pages", icon: FileInput, estimate: 400 },
  { id: "preprocess", label: "Preprocess", detail: "deskew · denoise · contrast", icon: Wand2, estimate: 700 },
  { id: "detect", label: "Detect", detail: "DBNet text lines", icon: ScanSearch, estimate: 900 },
  { id: "recognition", label: "Recognize", detail: "deep-learning model", icon: Type, estimate: Infinity },
  { id: "layout", label: "Layout", detail: "order · tables · forms", icon: LayoutPanelTop, estimate: 300 },
  { id: "postprocess", label: "Post-process", detail: "error correction", icon: SpellCheck2, estimate: 300 },
] as const;

interface Props {
  running: boolean;
  timings?: Record<string, number>;
  engineLabel?: string;
}

export function PipelineViz({ running, timings, engineLabel }: Props) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed(performance.now() - start), 80);
    return () => window.clearInterval(timer);
  }, [running]);

  // While running, walk through the stages with rough estimates; recognition holds until done.
  let active = 0;
  if (running) {
    let budget = elapsed;
    for (let index = 0; index < STAGES.length; index += 1) {
      budget -= STAGES[index].estimate;
      if (budget < 0) {
        active = index;
        break;
      }
    }
  }
  const done = !running && Boolean(timings);
  const total = timings?.total ?? 0;
  const stageMs = (id: string) => {
    if (!timings) return undefined;
    if (id === "detect") return undefined; // detection time is part of "recognition" for two-stage engines
    return timings[id];
  };

  return (
    <div className={`pipeline ${running ? "running" : ""} ${done ? "done" : ""}`} role="status" aria-live="polite">
      <div className="pipeline-head">
        <span className="section-label">{running ? "Pipeline running" : "Pipeline complete"}</span>
        <span className="pipeline-clock">
          {running ? `${(elapsed / 1000).toFixed(1)} s` : ms(total)}
          {engineLabel && <em> · {engineLabel}</em>}
        </span>
      </div>
      <ol className="pipeline-track">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon;
          const state = done ? "complete" : running ? (index < active ? "complete" : index === active ? "active" : "waiting") : "waiting";
          const value = stageMs(stage.id);
          return (
            <li key={stage.id} className={`stage ${state}`}>
              <motion.span
                className="stage-node"
                animate={state === "active" ? { scale: [1, 1.08, 1] } : { scale: 1 }}
                transition={state === "active" ? { repeat: Infinity, duration: 1.2 } : undefined}
              >
                {state === "complete" ? <Check size={16} /> : <Icon size={16} />}
              </motion.span>
              <span className="stage-copy">
                <strong>{stage.label}</strong>
                <small>{done && value !== undefined ? ms(value) : stage.detail}</small>
              </span>
              {index < STAGES.length - 1 && <span className="stage-beam" aria-hidden="true"><i /></span>}
            </li>
          );
        })}
      </ol>
      {done && timings && total > 0 && (
        <div className="timing-bar" aria-label="Where the time went">
          {STAGES.filter((stage) => stage.id !== "detect").map((stage) => {
            const value = timings[stage.id] ?? 0;
            const share = (value / total) * 100;
            return share > 0.5 ? (
              <motion.span
                key={stage.id}
                className={`timing-seg seg-${stage.id}`}
                initial={{ width: 0 }}
                animate={{ width: `${share}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                title={`${stage.label}: ${ms(value)} (${share.toFixed(0)}%)`}
              />
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}
