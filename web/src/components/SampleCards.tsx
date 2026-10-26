import { Loader2, Zap } from "lucide-react";
import type { SampleDoc } from "../types";
import { shortName } from "../lib/engines";

interface Props {
  samples: SampleDoc[];
  loadingId: string | null;
  disabled?: boolean;
  onPick: (sample: SampleDoc) => void;
}

export function SampleCards({ samples, loadingId, disabled, onPick }: Props) {
  if (!samples.length) return null;
  return (
    <section className="samples" aria-label="Sample documents">
      <div className="sec-head"><h3>Or try a sample document</h3><span className="muted">Runs instantly with the recommended settings</span></div>
      <div className="sample-grid">
        {samples.map((sample) => (
          <button key={sample.id} type="button" className="sample card" disabled={disabled} onClick={() => onPick(sample)}>
            <span className="sample-thumb"><img src={`/api/v1/samples/${encodeURIComponent(sample.id)}/thumb`} alt="" loading="lazy" /></span>
            <span className="sample-body">
              <b>{sample.title}</b>
              {sample.subtitle && <span>{sample.subtitle}</span>}
              <span className="badge outline">
                {loadingId === sample.id ? <Loader2 size={12} className="spin" /> : <Zap size={12} />}
                {sample.suggested_engine ? shortName(sample.suggested_engine) : "Recommended"}
              </span>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
