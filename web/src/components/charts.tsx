import { motion } from "framer-motion";
import { useState } from "react";

export const SERIES_COLORS: Record<string, string> = {
  paddle: "#94a3b8",
  cnn_bilstm: "#22d3ee",
  crnn: "#a78bfa",
  trocr: "#a3e635",
  trocr_ft: "#facc15",
  cnn_bilstm_pp: "#67e8f9",
  crnn_pp: "#c4b5fd",
};
export const colorFor = (key: string, index = 0) =>
  SERIES_COLORS[key] ?? ["#f472b6", "#fbbf24", "#34d399", "#60a5fa"][index % 4];

/* ---------------------------------------------------------------- Gauge */
export function Gauge({ value, size = 64, label }: { value: number; size?: number; label?: string }) {
  const radius = size / 2 - 6;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, value));
  const hue = 8 + clamped * 137;
  return (
    <svg className="gauge" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={label ?? `${Math.round(clamped * 100)}%`}>
      <circle cx={size / 2} cy={size / 2} r={radius} className="gauge-track" />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke={`hsl(${hue}, 90%, 60%)`}
        className="gauge-value"
        strokeDasharray={circumference}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: circumference * (1 - clamped) }}
        transition={{ duration: 1.1, ease: "easeOut" }}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="54%" textAnchor="middle" className="gauge-text">{Math.round(clamped * 100)}</text>
    </svg>
  );
}

/* ------------------------------------------------------------ Histogram */
export function Histogram({ bins, height = 44 }: { bins: number[]; height?: number }) {
  const max = Math.max(1, ...bins);
  return (
    <div className="histogram" style={{ height }} role="img" aria-label="Confidence distribution">
      {bins.map((count, index) => (
        <motion.span
          key={index}
          className="hist-bar"
          style={{ background: `hsl(${8 + (index / 9) * 137}, 85%, 58%)` }}
          initial={{ height: 0 }}
          animate={{ height: `${Math.max(4, (count / max) * 100)}%` }}
          transition={{ delay: index * 0.04, duration: 0.5 }}
          title={`${index * 10}–${index * 10 + 10}% confidence: ${count}`}
        />
      ))}
    </div>
  );
}

/* --------------------------------------------------------- Grouped bars */
export interface BarGroup {
  label: string;
  values: { key: string; label: string; value: number | null }[];
}

export function GroupedBars({ groups, unit = "%", lowerIsBetter = true, max }: {
  groups: BarGroup[];
  unit?: string;
  lowerIsBetter?: boolean;
  max?: number;
}) {
  const ceiling = max ?? Math.max(1, ...groups.flatMap((group) => group.values.map((v) => v.value ?? 0))) * 1.12;
  return (
    <div className="bars">
      {groups.map((group) => {
        const present = group.values.filter((v) => v.value !== null) as { key: string; label: string; value: number }[];
        const best = present.length
          ? (lowerIsBetter ? Math.min : Math.max)(...present.map((v) => v.value))
          : null;
        return (
          <div key={group.label} className="bar-group">
            <span className="bar-group-label">{group.label}</span>
            {group.values.map((item, index) => (
              <div key={item.key} className="bar-row">
                <span className="bar-name">{item.label}</span>
                <span className="bar-track">
                  {item.value !== null && (
                    <motion.span
                      className={`bar-fill ${item.value === best ? "best" : ""}`}
                      style={{ background: colorFor(item.key, index) }}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(item.value / ceiling) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.9, delay: index * 0.08, ease: "easeOut" }}
                    />
                  )}
                </span>
                <span className="bar-value">
                  {item.value === null ? "—" : `${item.value.toFixed(item.value < 10 ? 2 : 1)}${unit}`}
                  {item.value === best && present.length > 1 && <b className="best-badge">best</b>}
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------ Line chart */
export interface Series {
  key: string;
  label: string;
  points: (number | null)[];
}

export function LineChart({ series, xLabels, height = 220, yFormat = (v: number) => v.toFixed(2), yLabel }: {
  series: Series[];
  xLabels: (string | number)[];
  height?: number;
  yFormat?: (value: number) => string;
  yLabel?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 640;
  const pad = { l: 46, r: 14, t: 14, b: 30 };
  const values = series.flatMap((s) => s.points.filter((p): p is number => p !== null && Number.isFinite(p)));
  if (!values.length || xLabels.length === 0) return <div className="chart-empty">No data yet</div>;
  const yMax = Math.max(...values) * 1.08 || 1;
  const yMin = Math.min(0, Math.min(...values));
  const n = Math.max(2, xLabels.length);
  const x = (i: number) => pad.l + (i / (n - 1)) * (width - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
  return (
    <div className="line-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={yLabel ?? "Line chart"} onMouseLeave={() => setHover(null)}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} className="grid" />
            <text x={pad.l - 8} y={y(tick) + 4} textAnchor="end" className="axis">{yFormat(tick)}</text>
          </g>
        ))}
        {xLabels.map((label, i) => (
          (xLabels.length <= 12 || i % Math.ceil(xLabels.length / 12) === 0) && (
            <text key={i} x={x(i)} y={height - 8} textAnchor="middle" className="axis">{label}</text>
          )
        ))}
        {series.map((s, index) => {
          const path = s.points
            .map((p, i) => (p === null || !Number.isFinite(p) ? null : `${x(i)},${y(p)}`))
            .filter(Boolean)
            .join(" L ");
          if (!path) return null;
          return (
            <g key={s.key}>
              <motion.path
                d={`M ${path}`}
                fill="none"
                stroke={colorFor(s.key, index)}
                strokeWidth={2.4}
                strokeLinecap="round"
                strokeLinejoin="round"
                initial={{ pathLength: 0 }}
                whileInView={{ pathLength: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 1.2, ease: "easeInOut" }}
              />
              {s.points.map((p, i) => (p === null || !Number.isFinite(p) ? null : (
                <circle key={i} cx={x(i)} cy={y(p)} r={hover === i ? 5 : 3} fill={colorFor(s.key, index)} />
              )))}
            </g>
          );
        })}
        {xLabels.map((_, i) => (
          <rect key={i} x={x(i) - (width / n) / 2} y={0} width={width / n} height={height} fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={height - pad.b} className="hover-line" />}
      </svg>
      <div className="chart-legend">
        {series.map((s, index) => (
          <span key={s.key}>
            <i style={{ background: colorFor(s.key, index) }} />
            {s.label}
            {hover !== null && s.points[hover] != null && <b>{yFormat(s.points[hover] as number)}</b>}
          </span>
        ))}
        {hover !== null && <span className="legend-x">@ {xLabels[hover]}</span>}
      </div>
    </div>
  );
}
