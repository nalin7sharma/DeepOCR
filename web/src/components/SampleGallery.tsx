import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import type { SampleDoc } from "../types";

interface Props {
  samples: SampleDoc[];
  loadingId: string | null;
  onPick: (sample: SampleDoc) => void;
}

export function SampleGallery({ samples, loadingId, onPick }: Props) {
  if (!samples.length) return null;
  return (
    <section className="samples" aria-label="Sample documents">
      <div className="section-label"><Sparkles size={14} /> One-click demos</div>
      <div className="sample-grid">
        {samples.map((sample, index) => (
          <motion.button
            key={sample.id}
            type="button"
            className={`sample-card ${loadingId === sample.id ? "loading" : ""}`}
            onClick={() => onPick(sample)}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 * index }}
            whileHover={{ y: -4 }}
          >
            <span className="sample-thumb">
              <img src={`/api/v1/samples/${encodeURIComponent(sample.id)}/thumb`} alt="" loading="lazy" />
              <span className="sample-kind">{sample.kind ?? "document"}</span>
            </span>
            <span className="sample-copy">
              <strong>{sample.title}</strong>
              {sample.subtitle && <small>{sample.subtitle}</small>}
            </span>
          </motion.button>
        ))}
      </div>
    </section>
  );
}
