import { motion } from "framer-motion";
import { Brain, Cpu, Layers, Sparkles } from "lucide-react";
import type { EngineInfo } from "../types";
import { compact, percent } from "../lib/text";

const ICONS: Record<string, typeof Cpu> = { paddle: Cpu, crnn: Layers, cnn_bilstm: Brain, trocr: Sparkles, trocr_ft: Sparkles };

function headlines(engine: EngineInfo): string[] {
  const iam = engine.metrics?.iam;
  const print = engine.metrics?.synthetic;
  const tags = [];
  if (iam) tags.push(`handwriting CER ${iam.cer.toFixed(1)}%`);
  if (print) tags.push(`print CER ${print.cer.toFixed(1)}%`);
  if (!tags.length && engine.info?.val_cer != null) tags.push(`val CER ${percent(engine.info.val_cer)}`);
  return tags;
}

interface Props {
  engines: EngineInfo[];
  value: string;
  onChange: (id: string) => void;
  multiple?: false;
}

interface MultiProps {
  engines: EngineInfo[];
  value: string[];
  onChange: (ids: string[]) => void;
  multiple: true;
}

export function EnginePicker(props: Props | MultiProps) {
  const { engines } = props;
  const selected = (id: string) => (props.multiple ? props.value.includes(id) : props.value === id);
  const toggle = (id: string) => {
    if (props.multiple) {
      props.onChange(props.value.includes(id) ? props.value.filter((item) => item !== id) : [...props.value, id]);
    } else {
      props.onChange(id);
    }
  };
  return (
    <div className="engine-grid" role={props.multiple ? "group" : "radiogroup"} aria-label="OCR engine">
      {engines.map((engine) => {
        const Icon = ICONS[engine.id] ?? Cpu;
        const active = selected(engine.id);
        const stats = headlines(engine);
        return (
          <motion.button
            key={engine.id}
            type="button"
            role={props.multiple ? "checkbox" : "radio"}
            aria-checked={active}
            disabled={!engine.available}
            className={`engine-card engine-${engine.id} ${active ? "active" : ""}`}
            onClick={() => toggle(engine.id)}
            whileHover={engine.available ? { y: -3 } : undefined}
            whileTap={engine.available ? { scale: 0.98 } : undefined}
            title={engine.error ?? engine.description}
          >
            {active && <motion.span layoutId={props.multiple ? undefined : "engine-glow"} className="engine-glow" />}
            <span className="engine-head">
              <span className="engine-icon"><Icon size={18} /></span>
              <span className="engine-family">{engine.family}</span>
            </span>
            <strong>{engine.label}</strong>
            <small>{engine.recognizer}</small>
            <span className="engine-foot">
              {!engine.available ? (
                <em className="tag warn">{engine.info?.training ? "training…" : "not available"}</em>
              ) : engine.info?.training ? (
                <em className="tag live">training · epoch {engine.info.epoch}</em>
              ) : engine.loaded ? (
                <em className="tag ok">ready</em>
              ) : (
                <em className="tag">loads on first use</em>
              )}
              {stats.map((stat) => <em key={stat} className="tag metric">{stat}</em>)}
              {engine.info?.parameters ? <em className="tag">{compact(engine.info.parameters)} params</em> : null}
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}
