import { motion } from "framer-motion";
import { AlertTriangle, Clock3, FileStack, Rows3, SpellCheck2, Type } from "lucide-react";
import { useCountUp } from "../hooks";
import type { OcrResponse } from "../types";
import { Gauge, Histogram } from "./charts";

function Counter({ value, decimals = 0, suffix = "" }: { value: number; decimals?: number; suffix?: string }) {
  const animated = useCountUp(value);
  return <>{animated.toFixed(decimals)}{suffix}</>;
}

export function StatsRibbon({ result }: { result: OcrResponse }) {
  const stats = result.stats;
  const seconds = (result.timings?.total ?? result.inference_ms) / 1000;
  const tiles = [
    { icon: FileStack, label: "Pages", value: stats?.pages ?? result.pages.length },
    { icon: Rows3, label: "Lines", value: stats?.lines ?? 0 },
    { icon: Type, label: "Words", value: stats?.words ?? 0 },
    { icon: AlertTriangle, label: "To review", value: stats?.review_items ?? 0, tone: (stats?.review_items ?? 0) > 0 ? "warn" : "ok" },
    { icon: SpellCheck2, label: "Auto-fixes", value: stats?.corrections ?? 0 },
  ];
  return (
    <motion.section className="stats-ribbon" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} aria-label="Extraction summary">
      <div className="stat-tile hero-stat">
        <Gauge value={stats?.mean_confidence ?? 0} size={68} label="Mean recognition confidence" />
        <div>
          <span className="stat-label">Mean confidence</span>
          <strong className="stat-sub">{result.model}</strong>
        </div>
      </div>
      <div className="stat-tile">
        <Clock3 size={16} />
        <span className="stat-label">Total time</span>
        <strong><Counter value={seconds} decimals={2} suffix=" s" /></strong>
      </div>
      {tiles.map(({ icon: Icon, label, value, tone }) => (
        <div key={label} className={`stat-tile ${tone ?? ""}`}>
          <Icon size={16} />
          <span className="stat-label">{label}</span>
          <strong><Counter value={value} /></strong>
        </div>
      ))}
      {stats && stats.regions > 0 && (
        <div className="stat-tile wide">
          <span className="stat-label">Confidence distribution · {stats.regions} regions</span>
          <Histogram bins={stats.confidence_histogram} />
        </div>
      )}
    </motion.section>
  );
}
