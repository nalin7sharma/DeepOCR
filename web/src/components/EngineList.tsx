import { Sparkles, Trophy, Zap } from "lucide-react";
import type { EngineInfo } from "../types";
import { ENGINE_BADGE, ENGINE_COPY, colorFor, msPerLine, shortName, sortEngines, speedLabel } from "../lib/engines";

function Meter({ value }: { value: number }) {
  const width = Math.max(4, Math.min(100, (value / 30) * 100));
  const tone = value < 10 ? "#17b26a" : value < 25 ? "#f79009" : "#f04438";
  return <span className="meter"><i style={{ width: `${width}%`, background: tone }} /></span>;
}

interface Props {
  engines: EngineInfo[];
  value: string;
  onChange: (id: string) => void;
}

/** Radio list of engines with each one's measured error rates and speed. */
export function EngineList({ engines, value, onChange }: Props) {
  return (
    <div className="eng-list" role="radiogroup" aria-label="Recognition engine">
      {sortEngines(engines).map((engine) => {
        const copy = ENGINE_COPY[engine.id] ?? { tag: engine.family, ours: false, desc: engine.description };
        const badge = ENGINE_BADGE[engine.id];
        const hw = engine.metrics?.iam?.cer;
        const pr = engine.metrics?.synthetic?.cer;
        const on = value === engine.id;
        return (
          <button
            key={engine.id}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={!engine.available}
            className={`eng ${on ? "on" : ""}`}
            onClick={() => onChange(engine.id)}
            title={engine.error ?? engine.description}
          >
            <span className="radio" />
            <span className="eng-id" style={{ ["--c" as string]: colorFor(engine.id) }} />
            <span className="eng-main">
              <span className="eng-name">
                {shortName(engine.id, engine.label)}
                {badge && (
                  <span className={`badge ${badge.tone}`}>
                    {badge.tone === "success" ? <Trophy size={12} /> : <Zap size={12} />}{badge.text}
                  </span>
                )}
                {!engine.available && <span className="badge warning">unavailable</span>}
              </span>
              <span className="eng-desc"><span className={`eng-tag ${copy.ours ? "ours" : ""}`}>{copy.tag}</span>{copy.desc}</span>
            </span>
            <span className="eng-metric">
              <span className="mlabel">Handwriting</span>
              <b className="num">{hw !== undefined ? `${hw.toFixed(1)}%` : "—"}</b>
              {hw !== undefined ? <Meter value={hw} /> : <span className="meter" />}
            </span>
            <span className="eng-metric print">
              <span className="mlabel">Print</span>
              <b className="num">{pr !== undefined ? `${pr.toFixed(1)}%` : "—"}</b>
              {pr !== undefined ? <Meter value={pr} /> : <span className="meter" />}
            </span>
            <span className="eng-speed num">{speedLabel(msPerLine(engine))}<br /><span className="mlabel">per line</span></span>
          </button>
        );
      })}
    </div>
  );
}

/** Compact multi-select chips (Compare page). */
export function EngineChips({ engines, value, onChange }: { engines: EngineInfo[]; value: string[]; onChange: (ids: string[]) => void }) {
  return (
    <div className="chips" role="group" aria-label="Engines to compare">
      <span className="label">Engines</span>
      {sortEngines(engines).map((engine) => {
        const on = value.includes(engine.id);
        return (
          <button
            key={engine.id}
            type="button"
            role="checkbox"
            aria-checked={on}
            disabled={!engine.available}
            className={`chip ${on ? "on" : ""}`}
            onClick={() => onChange(on ? value.filter((id) => id !== engine.id) : [...value, engine.id])}
          >
            <i style={{ background: colorFor(engine.id) }} />{shortName(engine.id, engine.label)}
            {engine.id === "ensemble" && <Sparkles size={12} />}
          </button>
        );
      })}
    </div>
  );
}
