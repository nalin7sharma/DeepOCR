import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { BarRow, LineChart, type Series } from "../components/charts";
import { Segmented } from "../components/Ui";
import { colorFor } from "../lib/engines";
import { charDiff } from "../lib/text";
import type { EvalSetMetrics, MetricsResponse } from "../types";

const LABELS: Record<string, string> = {
  ensemble: "Consensus vote",
  trocr_ft: "TrOCR · fine-tuned",
  trocr: "TrOCR (pretrained)",
  crnn: "CRNN",
  cnn_bilstm: "CNN-BiLSTM",
  paddle: "PaddleOCR",
  crnn_pp: "CRNN + correction",
  cnn_bilstm_pp: "CNN-BiLSTM + correction",
};
const TABLE_LABELS: Record<string, string> = {
  ensemble: "Consensus vote (5 models)",
  trocr_ft: "TrOCR · fine-tuned (ours)",
  trocr: "TrOCR (pretrained)",
  crnn: "CRNN (ours)",
  cnn_bilstm: "CNN-BiLSTM (ours, baseline)",
  paddle: "PaddleOCR (reference)",
  crnn_pp: "CRNN + error correction",
  cnn_bilstm_pp: "CNN-BiLSTM + error correction",
};
const TABLE_ORDER = ["cnn_bilstm", "cnn_bilstm_pp", "crnn", "crnn_pp", "trocr", "trocr_ft", "paddle", "ensemble"];
const BAR_MODELS = ["ensemble", "trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"];
const ROBUST_ORDER = ["cnn_bilstm", "crnn", "trocr", "trocr_ft", "paddle"];

const SECTIONS = [
  ["overview", "Overview"],
  ["comparison", "Model comparison"],
  ["voting", "Model voting"],
  ["training", "Training"],
  ["robustness", "Robustness"],
  ["predictions", "Sample predictions"],
  ["architectures", "Architectures"],
  ["datasets", "Datasets"],
] as const;

type Metric = "cer" | "wer" | "accuracy" | "speed";
const METRIC_LABEL: Record<Metric, string> = { cer: "character error rate", wer: "word error rate", accuracy: "lines read perfectly", speed: "milliseconds per line" };

const ARCHITECTURES = [
  {
    id: "cnn_bilstm",
    name: "CNN + BiLSTM + CTC",
    role: "Baseline · trained from scratch",
    blocks: ["Line image 32×512", "4 conv blocks, 32→256 channels", "Map-to-sequence (128 steps)", "2× BiLSTM, 192 units", "Linear + softmax", "CTC decoding"],
    note: "The CNN extracts features column by column, the BiLSTM reads them in both directions, and CTC aligns characters without per-character labels.",
  },
  {
    id: "crnn",
    name: "CRNN (Shi et al., 2016)",
    role: "Model 2 · trained from scratch",
    blocks: ["Line image 32×512", "7-layer VGG CNN + batch norm", "Height-only pooling (2×1)", "BiLSTM → BiLSTM", "CTC loss", "Greedy / beam decoding"],
    note: "A deeper VGG backbone whose pooling shrinks only the height, so long handwritten lines keep their horizontal resolution.",
  },
  {
    id: "trocr_ft",
    name: "TrOCR (ViT encoder + Transformer decoder)",
    role: "Advanced · transfer learning, fine-tuned",
    blocks: ["Image 384×384 RGB", "16×16 patches", "DeiT encoder (self-attention)", "Cross-attention", "Autoregressive decoder", "Word-piece tokens"],
    note: "Self-attention captures long-range context. Fine-tuning on our IAM + synthetic data cut its printed-text error from 20.4% to 2.6%.",
  },
];

/** Mean time per line across the test sets, as in the comparison table. */
function avgMs(models: Record<string, Record<string, EvalSetMetrics>>, id: string): number | null {
  const sets = Object.values(models[id] ?? {}).filter((m) => typeof m?.ms_per_line === "number");
  return sets.length ? sets.reduce((sum, m) => sum + m.ms_per_line, 0) / sets.length : null;
}

const fmtPct = (value: number) => `${value.toFixed(1)}%`;
const fmtMs = (value: number) => (value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`);

function useActiveSection(ids: readonly string[]) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin: "-80px 0px -60% 0px" });
    ids.forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

function Section({ id, title, note, actions, children }: { id: string; title: string; note?: string; actions?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section id={id} className="card rsec">
      <div className="rsec-h">
        <h2>{title}</h2>
        {actions}
        {note && !actions && <span className="muted">{note}</span>}
      </div>
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ comparison */
function Comparison({ models }: { models: Record<string, Record<string, EvalSetMetrics>> }) {
  const [metric, setMetric] = useState<Metric>("cer");
  const value = (id: string, set: string) => {
    const m = models[id]?.[set];
    if (!m) return null;
    return metric === "speed" ? m.ms_per_line : metric === "accuracy" ? m.accuracy : m[metric];
  };
  const chart = (set: string, title: string) => {
    const rows = BAR_MODELS.map((id) => ({ id, v: value(id, set) })).filter((row): row is { id: string; v: number } => row.v !== null);
    rows.sort((a, b) => (metric === "accuracy" ? b.v - a.v : a.v - b.v));
    const max = Math.max(...rows.map((row) => row.v));
    return (
      <div>
        <span className="ch-title">{title} · {METRIC_LABEL[metric]}</span>
        {rows.map((row) => (
          <BarRow key={row.id} label={LABELS[row.id]} value={row.v} max={max} color={colorFor(row.id)} format={metric === "speed" ? fmtMs : fmtPct} strong={row.id === "ensemble"} />
        ))}
      </div>
    );
  };
  const cell = (id: string, set: string, key: "cer" | "wer") => {
    const v = models[id]?.[set]?.[key];
    return v === undefined ? "—" : v.toFixed(2);
  };
  const speed = (id: string) => {
    const value = avgMs(models, id);
    return value === null ? "—" : Math.round(value);
  };
  return (
    <Section
      id="comparison"
      title="Model comparison"
      actions={(
        <Segmented<Metric>
          small
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[{ value: "cer", label: "CER" }, { value: "wer", label: "WER" }, { value: "accuracy", label: "Accuracy" }, { value: "speed", label: "Speed" }]}
        />
      )}
    >
      <div className="two">
        {chart("iam", "IAM handwriting")}
        {chart("synthetic", "Printed & degraded")}
      </div>
      <div className="rt-wrap" style={{ marginTop: 18 }}>
        <table className="rt">
          <thead>
            <tr><th>Model</th><th className="n">Handwriting CER %</th><th className="n">WER %</th><th className="n">Print CER %</th><th className="n">WER %</th><th className="n">ms / line</th></tr>
          </thead>
          <tbody>
            {TABLE_ORDER.filter((id) => models[id]).map((id) => (
              <tr key={id} className={id === "ensemble" ? "hl" : ""}>
                <td><span className="dotc" style={{ background: colorFor(id) }} />{id.endsWith("_pp") ? <span className="muted">{TABLE_LABELS[id]}</span> : <b>{TABLE_LABELS[id] ?? id}</b>}</td>
                <td className="n">{cell(id, "iam", "cer")}</td>
                <td className="n">{cell(id, "iam", "wer")}</td>
                <td className="n">{cell(id, "synthetic", "cer")}</td>
                <td className="n">{cell(id, "synthetic", "wer")}</td>
                <td className="n">{speed(id)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="notes">
        <li>Held-out test sets: 600 IAM lines from writers never seen in training, and 600 printed or degraded lines. Lower CER and WER are better.</li>
        <li>“+ error correction” rows apply our dictionary and character-confusion corrector to the CTC models’ output.</li>
        <li>Times are per line on a laptop CPU (Intel i7-1255U, no GPU). The consensus vote runs all five models.</li>
      </ul>
    </Section>
  );
}

/* ------------------------------------------------------------------ voting */
function Voting({ metrics }: { metrics: MetricsResponse }) {
  const ensemble = metrics.ensemble;
  if (!ensemble) {
    return <Section id="voting" title="Model voting"><p className="empty">Run <code className="code">python scripts/ensemble_eval.py</code> to measure the consensus vote.</p></Section>;
  }
  const members = Object.keys(ensemble.weights);
  const SET_LABELS: Record<string, string> = { iam: "Handwriting", synthetic: "Print" };
  const sets = ["iam", "synthetic"].filter((set) => ensemble.sets[set]?.ensemble_cv && members.some((id) => ensemble.sets[set]?.[id]));
  const bestSingle = (set: string) => {
    const rows = members.map((id) => ensemble.sets[set]?.[id]).filter(Boolean);
    return rows.reduce((a, b) => (a.cer <= b.cer ? a : b));
  };
  const row = (set: string, key: string) => ensemble.sets[set]?.[key];
  const gain = (set: string, metric: "cer" | "wer") => {
    const single = bestSingle(set)[metric];
    const vote = row(set, "ensemble_cv")?.[metric] ?? single;
    return single > 0 ? Math.round(((vote - single) / single) * 100) : 0;
  };
  const weights = Object.entries(ensemble.weights).sort((a, b) => b[1] - a[1]).map(([id, w]) => `${LABELS[id]?.replace(" (pretrained)", "") ?? id} ${w}`).join(", ");
  const example = [["trocr_ft", "fan"], ["crnn", "fan"], ["trocr", "Pan"], ["paddle", "Pan"], ["cnn_bilstm", "Pan"]];
  return (
    <Section id="voting" title="Model voting" note={`2-fold cross-validated on ${ensemble.lines.toLocaleString()} test lines`}>
      <div className="vote-grid">
        <div className="vote-explain">
          <p>
            Every model reads each line. The readings are aligned word by word, and each word is decided by a weighted vote,
            where a vote counts <b>model weight × that model’s confidence</b>. A word several models agree on beats one model’s guess,
            and words without a majority are flagged for review in the Studio.
          </p>
          <div className="vote-example">
            <span className="label">Real example: “Tin ___ Alley”, five readings</span>
            {example.map(([id, word]) => (
              <div key={id} className="ve"><span><i className="dotc" style={{ background: colorFor(id), display: "inline-block", width: 8, height: 8, borderRadius: 4, marginRight: 8 }} />{LABELS[id]}</span><b>{word}</b></div>
            ))}
            <div className="ve win"><span>Vote result (correct)</span><b>Pan · 66% of the vote</b></div>
          </div>
          <p className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>Weights tuned on held-out lines: {weights}.</p>
        </div>
        <div className="rt-wrap">
          <table className="rt">
            <thead><tr><th />{sets.flatMap((set) => [<th key={`${set}c`} className="n">{SET_LABELS[set]} CER</th>, <th key={`${set}w`} className="n">WER</th>])}</tr></thead>
            <tbody>
              <tr>
                <td>Best single model</td>
                {sets.flatMap((set) => [<td key={`${set}c`} className="n">{bestSingle(set).cer.toFixed(2)}%</td>, <td key={`${set}w`} className="n">{bestSingle(set).wer.toFixed(2)}%</td>])}
              </tr>
              <tr className="hl">
                <td><b>Consensus vote</b></td>
                {sets.flatMap((set) => [<td key={`${set}c`} className="n"><b>{row(set, "ensemble_cv")?.cer.toFixed(2)}%</b></td>, <td key={`${set}w`} className="n"><b>{row(set, "ensemble_cv")?.wer.toFixed(2)}%</b></td>])}
              </tr>
              <tr>
                <td className="muted">Ceiling (best model picked per line)</td>
                {sets.flatMap((set) => [<td key={`${set}c`} className="n muted">{row(set, "oracle_best_line")?.cer.toFixed(2)}%</td>, <td key={`${set}w`} className="n muted">{row(set, "oracle_best_line")?.wer.toFixed(2)}%</td>])}
              </tr>
              <tr className="gain">
                <td>Fewer errors with voting</td>
                {sets.flatMap((set) => [<td key={`${set}c`} className="n">{gain(set, "cer")}%</td>, <td key={`${set}w`} className="n">{gain(set, "wer")}%</td>])}
              </tr>
            </tbody>
          </table>
          <ul className="notes">
            <li>The ceiling is an oracle that knows the right answer; it shows how much room voting still has.</li>
            <li>Weights are tuned on one half of the test lines and scored on the other half, then the halves swap.</li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ training */
function Training({ metrics }: { metrics: MetricsResponse }) {
  const curves: Series[] = ["cnn_bilstm", "crnn"].flatMap((id) => {
    const run = metrics.training[id];
    if (!run) return [];
    return [{ id, label: LABELS[id], color: colorFor(id), points: run.history.val_cer.map((v, i) => ({ x: i + 1, y: v * 100 })) }];
  });
  const ft = metrics.training.trocr_ft;
  const steps = ft?.history.step ?? [];
  const ftSeries: Series[] = ft ? [
    { id: "iam", label: "Handwriting", color: colorFor("trocr"), points: (ft.history.val_cer_iam ?? []).map((v, i) => ({ x: steps[i] ?? i, y: v })) },
    { id: "synthetic", label: "Print", color: colorFor("trocr_ft"), points: (ft.history.val_cer_synthetic ?? []).map((v, i) => ({ x: steps[i] ?? i, y: v })) },
  ] : [];
  const print = ft?.history.val_cer_synthetic ?? [];
  return (
    <div className="two" id="training" style={{ scrollMarginTop: 80 }}>
      <Section id="training-curves" title="Training" note="Validation CER per epoch">
        {curves.length ? (
          <>
            <LineChart label="Validation character error rate per epoch" series={curves} xFormat={(x) => `Epoch ${x}`} yFormat={(y) => `${y}%`} />
            <div className="lgd">
              {curves.map((s) => (
                <span key={s.id}><i style={{ background: s.color }} />{s.label} · best <b>{((metrics.training[s.id].best_val_cer ?? 0) * 100).toFixed(1)}%</b></span>
              ))}
            </div>
          </>
        ) : <p className="empty">No training history found.</p>}
      </Section>
      <Section id="finetune" title="Fine-tuning TrOCR" note="Validation CER by step">
        {ftSeries.length ? (
          <>
            <LineChart label="TrOCR fine-tuning validation error by step" series={ftSeries} xFormat={(x) => `Step ${x}`} yFormat={(y) => `${y}%`} />
            <div className="lgd">
              <span><i style={{ background: ftSeries[0].color }} />Handwriting</span>
              <span><i style={{ background: ftSeries[1].color }} />Print{print.length > 1 && <>: <b>{print[0].toFixed(1)}% → {Math.min(...print).toFixed(1)}%</b></>}</span>
            </div>
          </>
        ) : <p className="empty">Run <code className="code">python scripts/finetune_trocr.py</code> to fine-tune TrOCR.</p>}
      </Section>
    </div>
  );
}

/* ------------------------------------------------------------------ robustness */
function Robustness({ metrics }: { metrics: MetricsResponse }) {
  const data = metrics.robustness;
  if (!data) return <Section id="robustness" title="Robustness"><p className="empty">Run <code className="code">python scripts/robustness.py</code> to generate this analysis.</p></Section>;
  const models = ROBUST_ORDER.filter((id) => data.models[id]);
  const top = Math.max(...models.flatMap((id) => data.conditions.flatMap((c) => data.models[id][c.id] ?? [])));
  return (
    <Section id="robustness" title="Robustness" note={`CER on ${data.samples} IAM lines as each degradation gets stronger`}>
      <div className="smgrid">
        {data.conditions.map((condition) => (
          <div key={condition.id} className="sm card">
            <b>{condition.label}</b>
            <LineChart
              compact
              height={150}
              label={`${condition.label}: CER by model`}
              yMax={top}
              xTicks={condition.levels.map((_, i) => i)}
              xFormat={(i) => String(condition.levels[i] ?? i)}
              yFormat={(y) => `${Math.round(y)}%`}
              series={models.map((id) => ({ id, label: LABELS[id], color: colorFor(id), points: (data.models[id][condition.id] ?? []).map((v, i) => ({ x: i, y: v })) }))}
            />
          </div>
        ))}
      </div>
      <div className="lgd">{models.map((id) => <span key={id}><i style={{ background: colorFor(id) }} />{LABELS[id]}</span>)}</div>
      <ul className="notes">
        <li>Rotation hurts every model; with our deskew step in front, error stays flat up to 8°.</li>
        <li>Noise, blur, low contrast and JPEG compression barely move the error, thanks to augmentation during training.</li>
      </ul>
    </Section>
  );
}

/* ------------------------------------------------------------------ predictions */
function Predictions({ metrics }: { metrics: MetricsResponse }) {
  const [set, setSet] = useState<"iam" | "synthetic">("iam");
  const [all, setAll] = useState(false);
  const samples = (metrics.samples ?? []).filter((sample) => sample.set === set);
  const shown = all ? samples : samples.slice(0, 4);
  const order = ["trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"];
  return (
    <Section
      id="predictions"
      title="Sample predictions"
      actions={<Segmented<"iam" | "synthetic"> small label="Test set" value={set} onChange={(value) => { setSet(value); setAll(false); }} options={[{ value: "iam", label: "Handwriting" }, { value: "synthetic", label: "Print" }]} />}
    >
      {!samples.length ? <p className="empty">Sample predictions appear after evaluation.</p> : (
        <>
          <div className="legend-row" style={{ marginTop: 0 }}>
            <span className="diff-ins">extra</span><span className="muted">characters the model added</span>
            <span className="diff-del">missing</span><span className="muted">characters it missed</span>
          </div>
          <div className="preds">
            {shown.map((sample, index) => (
              <div key={index} className="pred">
                <div className="pred-img">
                  <img src={`data:image/png;base64,${sample.image}`} alt={`Test line: ${sample.ground_truth}`} loading="lazy" />
                </div>
                <div className="pred-rows">
                  <div className="pred-row gt"><span>Ground truth</span><p>{sample.ground_truth}</p><em /></div>
                  {order.filter((id) => sample.predictions[id]).map((id) => (
                    <div key={id} className="pred-row">
                      <span><span className="dotc" style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, marginRight: 6, background: colorFor(id) }} />{LABELS[id]}</span>
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
          {samples.length > 4 && (
            <button type="button" className="btn sm" style={{ marginTop: 12 }} onClick={() => setAll(!all)}>{all ? "Show fewer" : `Show all ${samples.length}`}</button>
          )}
        </>
      )}
    </Section>
  );
}

/* ------------------------------------------------------------------ page */
export function Research({ metrics, error }: { metrics: MetricsResponse | null; error: string }) {
  const ids = useMemo(() => SECTIONS.map(([id]) => id), []);
  const active = useActiveSection(ids);
  const models = (metrics?.evaluation?.models ?? {}) as Record<string, Record<string, EvalSetMetrics>>;

  const kpis = useMemo(() => {
    const ids = Object.keys(models).filter((id) => !id.endsWith("_pp"));
    const best = (set: string) => ids.filter((id) => models[id]?.[set]).reduce<string | null>((a, b) => (a === null || models[b][set].cer < models[a][set].cer ? b : a), null);
    const fastest = ids.filter((id) => avgMs(models, id) !== null).reduce<string | null>((a, b) => (a === null || avgMs(models, b)! < avgMs(models, a)! ? b : a), null);
    const rotation = metrics?.robustness?.models.cnn_bilstm;
    return { hw: best("iam"), print: best("synthetic"), fastest, rotation };
  }, [models, metrics]);

  if (!metrics) {
    return (
      <div className="page">
        <div className="ph"><h1>Experiments &amp; results</h1></div>
        {error ? <div className="alert"><AlertCircle size={16} />Could not load the results: {error}</div> : <p className="empty"><Loader2 size={18} className="spin" style={{ display: "inline", verticalAlign: -3 }} /> Loading results…</p>}
      </div>
    );
  }

  const describe = (id: string | null) => (id === "ensemble" ? "consensus vote of 5 models" : id ? LABELS[id] : "");
  const rotation = kpis.rotation;

  return (
    <div className="page rpage">
      <nav className="toc" aria-label="On this page">
        <span className="label">On this page</span>
        {SECTIONS.map(([id, label]) => (
          <a
            key={id}
            href={`#/research`}
            className={active === id || (id === "training" && active === "training-curves") ? "on" : ""}
            onClick={(event) => {
              event.preventDefault();
              document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            {label}
          </a>
        ))}
      </nav>
      <div style={{ minWidth: 0 }}>
        <div className="ph" id="overview" style={{ scrollMarginTop: 80 }}>
          <h1>Experiments &amp; results</h1>
          <p>
            Every number comes from held-out test sets: 600 lines of IAM handwriting from writers never seen in training, and 600 printed
            or degraded lines. Measured on a laptop CPU.
          </p>
        </div>
        <div className="kpis">
          <div className="card kpi"><span className="label">Best on handwriting</span><b>{kpis.hw ? `${models[kpis.hw].iam.cer.toFixed(2)}%` : "—"}</b><span>CER · {describe(kpis.hw)}</span></div>
          <div className="card kpi"><span className="label">Best on print</span><b>{kpis.print ? `${models[kpis.print].synthetic.cer.toFixed(2)}%` : "—"}</b><span>CER · {describe(kpis.print)}</span></div>
          <div className="card kpi"><span className="label">Fastest</span><b>{kpis.fastest ? fmtMs(avgMs(models, kpis.fastest)!) : "—"}</b><span>per line · {describe(kpis.fastest)}</span></div>
          <div className="card kpi">
            <span className="label">Deskew at 8° tilt</span>
            <b>{rotation ? `${Math.round(rotation.rotation.at(-1) ?? 0)}→${Math.round(rotation.rotation_deskew.at(-1) ?? 0)}%` : "—"}</b>
            <span>CER · CNN-BiLSTM, without → with deskew</span>
          </div>
        </div>

        {Object.keys(models).length ? <Comparison models={models} /> : <Section id="comparison" title="Model comparison"><p className="empty">Run <code className="code">python scripts/evaluate.py</code> to score the models.</p></Section>}
        <Voting metrics={metrics} />
        <Training metrics={metrics} />
        <Robustness metrics={metrics} />
        <Predictions metrics={metrics} />

        <Section id="architectures" title="Architectures" note="From the baseline to the advanced model">
          <div className="arch-grid">
            {ARCHITECTURES.map((arch) => (
              <div key={arch.id} className="arch" style={{ ["--c" as string]: colorFor(arch.id) }}>
                <span className="badge outline">{arch.role}</span>
                <h3>{arch.name}</h3>
                <ol>{arch.blocks.map((block) => <li key={block}>{block}</li>)}</ol>
                <p>{arch.note}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="datasets" title="Datasets" note="Real handwriting plus synthetic print, with fixed splits">
          <div className="data-grid">
            <div className="dset">
              <span className="badge accent">Real handwriting</span>
              <h3>IAM Handwriting (lines)</h3>
              <p>English handwriting from 657 writers. The official writer-independent split means test writers are never seen in training.</p>
              <div className="splits"><span><b>Train</b>6,482</span><span><b>Val</b>976</span><span><b>Test</b>2,915</span></div>
            </div>
            <div className="dset">
              <span className="badge accent">Generated</span>
              <h3>Synthetic printed + script</h3>
              <p>Lines rendered with 19 printed and 13 handwriting-style fonts, then degraded with noise, blur, skew, JPEG, uneven lighting, ink bleed and inversion.</p>
              <div className="splits"><span><b>Train</b>5,005</span><span><b>Val</b>520</span><span><b>Test</b>975</span></div>
            </div>
            <div className="dset">
              <span className="badge accent">Training only</span>
              <h3>Augmentation</h3>
              <p>Applied on the fly: random rotation, brightness and contrast, Gaussian noise, blur, scaling and perspective warps. Test images are never augmented.</p>
              <div className="chips" style={{ marginTop: 12 }}>
                {["rotation", "brightness", "contrast", "noise", "blur", "scale", "perspective"].map((name) => <span key={name} className="chip on">{name}</span>)}
              </div>
            </div>
          </div>
        </Section>
        <p className="muted" style={{ fontSize: 12.5 }}>
          Confidence scores in the Studio are model probabilities, not accuracy. Accuracy claims come only from these labelled test sets.
        </p>
      </div>
    </div>
  );
}
