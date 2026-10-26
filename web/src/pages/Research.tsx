import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, BarChart3, Database, FlaskConical, Images, Network, ShieldAlert } from "lucide-react";
import { GroupedBars, LineChart, colorFor, type BarGroup } from "../components/charts";
import type { EnginesResponse, MetricsResponse } from "../types";
import { charDiff, compact } from "../lib/text";

const MODEL_ORDER = ["cnn_bilstm", "cnn_bilstm_pp", "crnn", "crnn_pp", "trocr", "trocr_ft", "paddle"];
const MODEL_LABELS: Record<string, string> = {
  cnn_bilstm: "CNN-BiLSTM (baseline)",
  cnn_bilstm_pp: "CNN-BiLSTM + post-processing",
  crnn: "CRNN",
  crnn_pp: "CRNN + post-processing",
  trocr: "TrOCR (zero-shot)",
  trocr_ft: "TrOCR (fine-tuned)",
  paddle: "PaddleOCR (reference)",
};

const ARCHITECTURES = [
  {
    id: "cnn_bilstm",
    name: "CNN + BiLSTM + CTC",
    role: "Baseline · trained from scratch",
    blocks: ["Line image 32×512", "4 conv blocks 32→256", "Map-to-sequence (T=128)", "2× BiLSTM 192", "Linear + softmax", "CTC decode"],
    note: "The CNN extracts visual features column by column, the BiLSTM reads them in both directions, and CTC aligns the characters without per-character labels.",
  },
  {
    id: "crnn",
    name: "CRNN (Shi et al., 2016)",
    role: "Model 2 · trained from scratch",
    blocks: ["Line image 32×512", "7-layer VGG CNN + BN", "Asymmetric pooling (2×1)", "BiLSTM → BiLSTM", "CTC loss", "Greedy / beam decode"],
    note: "A deeper VGG-style backbone with height-only pooling keeps horizontal resolution for long handwritten lines.",
  },
  {
    id: "trocr",
    name: "TrOCR (ViT encoder + Transformer decoder)",
    role: "Advanced model · transfer learning",
    blocks: ["Image 384×384 RGB", "16×16 patches", "ViT / DeiT encoder", "Cross-attention", "Autoregressive decoder", "Word-piece tokens"],
    note: "Self-attention captures long-range dependencies. We fine-tuned the pretrained model for 300 steps on our IAM + synthetic data; its printed-text CER fell from 20.4% to 2.6%.",
  },
];

function Section({ icon: Icon, title, subtitle, children }: { icon: typeof Activity; title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <motion.section className="glass research-section" initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }}>
      <header className="section-head">
        <span className="section-icon"><Icon size={18} /></span>
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </header>
      {children}
    </motion.section>
  );
}

export function Research({ metrics, engines, error }: { metrics: MetricsResponse | null; engines: EnginesResponse | null; error: string }) {
  const evaluation = metrics?.evaluation;
  const [metric, setMetric] = useState<"cer" | "wer" | "accuracy" | "ms_per_line">("cer");
  const models = useMemo(
    () => MODEL_ORDER.filter((id) => evaluation?.models?.[id]),
    [evaluation],
  );
  const testSets = Object.entries(evaluation?.test_sets ?? {});

  const groups: BarGroup[] = testSets.map(([setId, set]) => ({
    label: `${set.label} · n=${set.samples}`,
    values: models.map((id) => ({
      key: id,
      label: MODEL_LABELS[id] ?? id,
      value: (evaluation?.models[id]?.[setId] as { [k: string]: number } | undefined)?.[metric] ?? null,
    })),
  }));

  const training = Object.values(metrics?.training ?? {})
    .filter((run) => run.model === "cnn_bilstm" || run.model === "crnn")
    .sort((a, b) => MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model));
  const finetune = metrics?.training?.trocr_ft as (typeof training)[number] & { history: { step?: number[]; val_cer_iam?: number[]; val_cer_synthetic?: number[] } } | undefined;
  const maxEpochs = Math.max(0, ...training.map((run) => run.history.val_cer.length));
  const epochs = Array.from({ length: maxEpochs }, (_, i) => i + 1);

  const best = (setId: string) => {
    let winner: { id: string; cer: number } | null = null;
    for (const id of models) {
      const value = (evaluation?.models[id]?.[setId] as { cer?: number } | undefined)?.cer;
      if (value !== undefined && (!winner || value < winner.cer)) winner = { id, cer: value };
    }
    return winner;
  };

  const robustness = metrics?.robustness;
  const datasetCounts = (training[0]?.meta ?? {}) as { train_samples?: number; val_samples?: number };

  return (
    <div className="research">
      <header className="page-head">
        <p className="eyebrow"><span className="pulse-dot" /> Experiments &amp; evaluation</p>
        <h1>How the models <span className="gradient-text">actually perform.</span></h1>
        <p className="hero-sub">
          Every number here comes from held-out test splits that were never used in training or model selection:
          the official IAM writer-independent test set (handwriting) and a synthetic printed/script test set.
          CER and WER are averaged per line; lower is better.
        </p>
        {error && !metrics && <div className="error-banner">Metrics unavailable: {error}</div>}
      </header>

      <div className="kpi-row">
        {testSets.map(([setId, set]) => {
          const winner = best(setId);
          return (
            <div key={setId} className="glass kpi">
              <span className="stat-label">Best CER · {set.label}</span>
              <strong>{winner ? `${winner.cer.toFixed(2)}%` : "—"}</strong>
              <small>{winner ? MODEL_LABELS[winner.id] : "awaiting evaluation"}</small>
            </div>
          );
        })}
        <div className="glass kpi">
          <span className="stat-label">Training lines</span>
          <strong>{compact(datasetCounts.train_samples ?? null)}</strong>
          <small>IAM handwriting + synthetic print</small>
        </div>
        <div className="glass kpi">
          <span className="stat-label">Models compared</span>
          <strong>{models.length || engines?.engines.length || 0}</strong>
          <small>baseline · CRNN · Transformer · reference</small>
        </div>
      </div>

      <Section icon={BarChart3} title="Model comparison" subtitle="The experimental comparison table from the proposal, filled with measured results.">
        {models.length === 0 ? (
          <p className="empty-note">No evaluation results yet. Run <code>python scripts/evaluate.py --all</code> after training.</p>
        ) : (
          <>
            <div className="segmented wide" role="radiogroup" aria-label="Metric">
              {([["cer", "CER %"], ["wer", "WER %"], ["accuracy", "Line accuracy %"], ["ms_per_line", "ms / line"]] as const).map(([id, label]) => (
                <button key={id} type="button" role="radio" aria-checked={metric === id} className={metric === id ? "on" : ""} onClick={() => setMetric(id)}>{label}</button>
              ))}
            </div>
            <GroupedBars groups={groups} unit={metric === "ms_per_line" ? " ms" : "%"} lowerIsBetter={metric !== "accuracy"} max={metric === "accuracy" ? 100 : undefined} />
            <div className="table-scroll">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Architecture</th>
                    {testSets.map(([setId, set]) => <th key={setId} colSpan={3}>{set.label}</th>)}
                    <th>Inference</th>
                  </tr>
                  <tr className="sub">
                    <th /><th />
                    {testSets.map(([setId]) => ["CER", "WER", "Acc"].map((h) => <th key={setId + h}>{h}</th>))}
                    <th>ms / line</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((id) => {
                    const row = evaluation!.models[id];
                    const info = (row._info ?? {}) as { architecture?: string; parameters?: number };
                    const speeds = testSets.map(([setId]) => (row[setId] as { ms_per_line?: number } | undefined)?.ms_per_line).filter((v): v is number => v !== undefined);
                    return (
                      <tr key={id}>
                        <td><i className="dot" style={{ background: colorFor(id) }} />{MODEL_LABELS[id] ?? id}</td>
                        <td className="muted">{info.architecture ?? "—"}{info.parameters ? ` · ${compact(info.parameters)}` : ""}</td>
                        {testSets.map(([setId]) => {
                          const m = row[setId] as { cer: number; wer: number; accuracy: number } | undefined;
                          return m ? [m.cer, m.wer, m.accuracy].map((v, i) => <td key={setId + i} className="num">{v.toFixed(2)}</td>) : [0, 1, 2].map((i) => <td key={setId + i}>—</td>);
                        })}
                        <td className="num">{speeds.length ? (speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {evaluation?.notes && <ul className="notes">{evaluation.notes.map((note) => <li key={note}>{note}</li>)}</ul>}
          </>
        )}
      </Section>

      <Section icon={Activity} title="Training curves" subtitle="Validation CER per epoch for the models trained from scratch. Updates live while training runs.">
        {training.length === 0 ? <p className="empty-note">No training runs recorded yet.</p> : (
          <div className="chart-pair">
            <div>
              <span className="chart-title">Validation CER</span>
              <LineChart
                series={training.map((run) => ({ key: run.model, label: MODEL_LABELS[run.model] ?? run.model, points: run.history.val_cer.map((v) => v * 100) }))}
                xLabels={epochs}
                yFormat={(v) => `${v.toFixed(0)}%`}
              />
            </div>
            <div>
              <span className="chart-title">Training loss (CTC)</span>
              <LineChart
                series={training.map((run) => ({ key: run.model, label: MODEL_LABELS[run.model] ?? run.model, points: run.history.train_loss }))}
                xLabels={epochs}
                yFormat={(v) => v.toFixed(2)}
              />
            </div>
            <div className="run-cards">
              {training.map((run) => (
                <div key={run.model} className="run-card" style={{ ["--accent" as string]: colorFor(run.model) }}>
                  <strong>{MODEL_LABELS[run.model] ?? run.model}</strong>
                  <span>{run.finished ? "finished" : <b className="live">● training</b>} · epoch {run.epoch}/{run.epochs_planned}</span>
                  <span>best val CER <b>{run.best_val_cer !== null ? `${(run.best_val_cer * 100).toFixed(2)}%` : "—"}</b></span>
                  <span>{Math.round(run.elapsed_seconds / 60)} min on CPU · {compact((run.meta as { parameters?: number }).parameters ?? null)} params</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      {finetune && (finetune.history.step?.length ?? 0) > 1 && (
        <Section icon={Activity} title="Transformer fine-tuning" subtitle="TrOCR adapted to our IAM + synthetic training data (transfer learning). Validation CER at each checkpoint; step 0 is the pretrained model.">
          <div className="chart-pair">
            <div>
              <LineChart
                series={[
                  { key: "trocr", label: "IAM handwriting (val)", points: finetune.history.val_cer_iam ?? [] },
                  { key: "trocr_ft", label: "Synthetic print (val)", points: finetune.history.val_cer_synthetic ?? [] },
                ]}
                xLabels={(finetune.history.step ?? []).map((step) => `step ${step}`)}
                yFormat={(v) => `${v.toFixed(1)}%`}
              />
            </div>
            <div className="run-cards single">
              <div className="run-card" style={{ ["--accent" as string]: colorFor("trocr_ft") }}>
                <strong>TrOCR fine-tuning</strong>
                <span>{finetune.finished ? "finished" : <b className="live">● training</b>} · {(finetune.meta as { steps?: number }).steps ?? 0} steps</span>
                <span>best mean val CER <b>{finetune.best_val_cer !== null ? `${(finetune.best_val_cer * 100).toFixed(2)}%` : "—"}</b></span>
                <span>{Math.round(finetune.elapsed_seconds / 60)} min on CPU · {compact((finetune.meta as { parameters?: number }).parameters ?? null)} params</span>
              </div>
            </div>
          </div>
        </Section>
      )}

      <Section icon={ShieldAlert} title="Robustness to real-world degradation" subtitle="CER on IAM test lines as controlled noise, blur, rotation and low contrast are added. This is the proposal's real-world robustness test.">
        {!robustness ? <p className="empty-note">Run <code>python scripts/robustness.py</code> to generate this analysis.</p> : (
          <div className="robust-grid">
            {robustness.conditions.map((condition) => (
              <div key={condition.id}>
                <span className="chart-title">{condition.label}</span>
                <LineChart
                  height={190}
                  series={Object.entries(robustness.models).map(([id, curves]) => ({ key: id, label: MODEL_LABELS[id] ?? id, points: curves[condition.id] ?? [] }))}
                  xLabels={condition.levels}
                  yFormat={(v) => `${v.toFixed(0)}%`}
                />
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section icon={Images} title="Sample predictions" subtitle="Random held-out test lines. Character errors are highlighted against the ground truth.">
        {!metrics?.samples?.length ? <p className="empty-note">Sample predictions appear after evaluation.</p> : (
          <div className="prediction-list">
            {metrics.samples.slice(0, 12).map((sample, index) => (
              <div key={index} className="prediction">
                <div className="prediction-image">
                  <img src={`data:image/png;base64,${sample.image}`} alt={`Test line: ${sample.ground_truth}`} loading="lazy" />
                  <span className="tag">{sample.set}</span>
                </div>
                <div className="prediction-rows">
                  <div className="pred-row gt"><span>Ground truth</span><p>{sample.ground_truth}</p></div>
                  {MODEL_ORDER.filter((id) => sample.predictions[id]).map((id) => (
                    <div key={id} className="pred-row">
                      <span><i className="dot" style={{ background: colorFor(id) }} />{MODEL_LABELS[id]}</span>
                      <p>
                        {charDiff(sample.predictions[id].text, sample.ground_truth).map((op, i) => (
                          op.type === "equal" ? <span key={i}>{op.text}</span>
                            : op.type === "insert" ? <span key={i} className="diff-ins">{op.text}</span>
                            : <span key={i} className="diff-del">{op.text}</span>
                        ))}
                      </p>
                      <em>{(sample.predictions[id].cer * 100).toFixed(0)}%</em>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section icon={Network} title="Architectures" subtitle="From the baseline to the advanced model.">
        <div className="arch-grid">
          {ARCHITECTURES.map((arch) => (
            <div key={arch.id} className="arch-card" style={{ ["--accent" as string]: colorFor(arch.id) }}>
              <span className="tag">{arch.role}</span>
              <h3>{arch.name}</h3>
              <ol className="arch-flow">
                {arch.blocks.map((block, index) => (
                  <motion.li key={block} initial={{ opacity: 0, x: -8 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.07 }}>
                    {block}
                  </motion.li>
                ))}
              </ol>
              <p>{arch.note}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={Database} title="Datasets" subtitle="Real handwriting plus synthetic print, with official splits.">
        <div className="dataset-grid">
          <div className="dataset-card">
            <h3>IAM Handwriting (lines)</h3>
            <p>657 writers of English handwritten forms. We use the official writer-independent partition, so test-set writers are never seen in training.</p>
            <div className="split-bars">
              {[["train", 6482], ["val", 976], ["test", 2915]].map(([name, count]) => (
                <span key={name as string} style={{ flex: count as number }}><b>{name}</b>{(count as number).toLocaleString()}</span>
              ))}
            </div>
          </div>
          <div className="dataset-card">
            <h3>Synthetic printed + script</h3>
            <p>Lines rendered with 19 printed and 13 handwriting-style fonts, then degraded with noise, blur, JPEG artefacts, skew, uneven lighting, ink bleed and inversion.</p>
            <div className="split-bars">
              {[["train", 5005], ["val", 520], ["test", 975]].map(([name, count]) => (
                <span key={name as string} style={{ flex: count as number }}><b>{name}</b>{(count as number).toLocaleString()}</span>
              ))}
            </div>
          </div>
          <div className="dataset-card">
            <h3>Augmentation (training only)</h3>
            <p>Random rotation, brightness and contrast changes, Gaussian noise, blur, scaling without cropping, and perspective warps, all applied on the fly.</p>
            <div className="aug-chips">{["rotation", "brightness", "contrast", "noise", "blur", "scale", "perspective"].map((a) => <span key={a} className="chip on">{a}</span>)}</div>
          </div>
        </div>
      </Section>
      <p className="legend-note"><FlaskConical size={13} /> Confidence scores shown in the Studio are model probabilities, not accuracy. Accuracy claims come only from these labelled test sets.</p>
    </div>
  );
}
