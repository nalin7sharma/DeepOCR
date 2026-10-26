import { useRef, useState } from "react";

export interface Series {
  id: string;
  label: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

interface Props {
  series: Series[];
  height?: number;
  /** Compact small-multiple: fewer ticks, no hover read-out. */
  compact?: boolean;
  xTicks?: number[];
  xFormat?: (x: number) => string;
  yFormat?: (y: number) => string;
  yMax?: number;
  label: string;
}

/** A round tick step (1, 2, 5 × 10ⁿ) that splits `range` into about `count` parts. */
const niceStep = (range: number, count: number) => {
  const raw = Math.max(range, 1e-9) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 5, 10].find((s) => s * magnitude >= raw) ?? 10) * magnitude;
};

/** Light line chart with a hover read-out of every series at the nearest x. */
export function LineChart({ series, height = 220, compact, xTicks, xFormat = String, yFormat = (y) => `${y}`, yMax, label }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const width = compact ? 300 : 560;
  const pad = compact ? { l: 30, r: 8, t: 8, b: 20 } : { l: 40, r: 12, t: 10, b: 26 };
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  if (!xs.length) return <p className="empty">No data yet.</p>;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const peak = yMax ?? Math.max(...ys);
  const step = niceStep(peak, compact ? 2 : 4);
  const top = Math.max(step, Math.ceil(peak / step - 1e-9) * step);
  const x = (value: number) => pad.l + ((value - xMin) / Math.max(1e-9, xMax - xMin)) * (width - pad.l - pad.r);
  const y = (value: number) => pad.t + (1 - Math.min(value, top) / top) * (height - pad.t - pad.b);
  const yTicks = Array.from({ length: Math.floor(top / step + 1e-9) + 1 }, (_, i) => i * step);
  const ticks = xTicks ?? [...new Set(xs)].sort((a, b) => a - b);
  const shownTicks = ticks.length > 8 ? ticks.filter((_, i) => i % Math.ceil(ticks.length / 7) === 0 || i === ticks.length - 1) : ticks;
  const allX = [...new Set(xs)].sort((a, b) => a - b);

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (compact || !ref.current) return;
    const bounds = ref.current.getBoundingClientRect();
    const px = ((event.clientX - bounds.left) / bounds.width) * width;
    const nearest = allX.reduce((best, value) => (Math.abs(x(value) - px) < Math.abs(x(best) - px) ? value : best), allX[0]);
    setHover(nearest);
  };

  return (
    <div style={{ position: "relative" }}>
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={label}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="g" x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} />
            <text className="ax" x={pad.l - 6} y={y(tick) + 4} textAnchor="end">{yFormat(+tick.toFixed(2))}</text>
          </g>
        ))}
        {shownTicks.map((tick) => (
          <text key={tick} className="ax" x={x(tick)} y={height - 6} textAnchor="middle">{xFormat(tick)}</text>
        ))}
        {hover !== null && <line className="hover-line" x1={x(hover)} x2={x(hover)} y1={pad.t} y2={height - pad.b} />}
        {series.map((s) => (
          <g key={s.id}>
            <polyline
              fill="none"
              stroke={s.color}
              strokeWidth={compact ? 2 : 2.25}
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeDasharray={s.dashed ? "5 4" : undefined}
              points={s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
            />
            {!compact && s.points.map((p) => (
              <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={hover === p.x ? 4.5 : 2.5} fill={hover === p.x ? s.color : "#fff"} stroke={s.color} strokeWidth={2} />
            ))}
          </g>
        ))}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: `${(x(hover) / width) * 100}%`, top: 4, transform: x(hover) > width * 0.6 ? "translateX(calc(-100% - 12px))" : "translateX(12px)" }}>
          <b>{xFormat(hover)}</b>
          {series.map((s) => {
            const point = s.points.find((p) => p.x === hover);
            return point ? (
              <span key={s.id} className="tip-sub" style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
                <span><i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: s.color, marginRight: 6 }} />{s.label}</span>
                <b className="num">{yFormat(+point.y.toFixed(2))}</b>
              </span>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}

/** Horizontal bar row used by the comparison charts. */
export function BarRow({ label, value, max, color, format, strong }: {
  label: string;
  value: number;
  max: number;
  color: string;
  format: (value: number) => string;
  strong?: boolean;
}) {
  return (
    <div className="brow">
      <span title={label} style={strong ? { fontWeight: 650 } : undefined}>{label}</span>
      <span className="btrack"><i style={{ width: `${Math.max(1.5, (value / Math.max(max, 1e-9)) * 100)}%`, background: color }} /></span>
      <b>{format(value)}</b>
    </div>
  );
}
