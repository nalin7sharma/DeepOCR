import { Contrast, Eraser, RotateCcw, SpellCheck2, SquareDashed } from "lucide-react";

const STEPS = [
  { id: "deskew", label: "Deskew", icon: RotateCcw, hint: "Straighten tilted scans" },
  { id: "denoise", label: "Denoise", icon: Eraser, hint: "Remove speckle noise" },
  { id: "contrast", label: "Contrast", icon: Contrast, hint: "CLAHE local contrast" },
  { id: "binarize", label: "Binarize", icon: SquareDashed, hint: "Adaptive threshold" },
];

interface Props {
  preprocess: string[];
  onPreprocess: (steps: string[]) => void;
  correct: boolean;
  onCorrect: (value: boolean) => void;
}

export function OptionsBar({ preprocess, onPreprocess, correct, onCorrect }: Props) {
  const toggle = (id: string) =>
    onPreprocess(preprocess.includes(id) ? preprocess.filter((step) => step !== id) : [...preprocess, id]);
  return (
    <div className="options-bar">
      <div className="option-group" role="group" aria-label="Preprocessing">
        <span className="option-label">Preprocess</span>
        {STEPS.map(({ id, label, icon: Icon, hint }) => (
          <button
            key={id}
            type="button"
            className={`chip ${preprocess.includes(id) ? "on" : ""}`}
            aria-pressed={preprocess.includes(id)}
            onClick={() => toggle(id)}
            title={hint}
          >
            <Icon size={14} /> {label}
          </button>
        ))}
      </div>
      <div className="option-group" role="group" aria-label="Post-processing">
        <span className="option-label">Post-process</span>
        <button
          type="button"
          className={`chip ${correct ? "on" : ""}`}
          aria-pressed={correct}
          onClick={() => onCorrect(!correct)}
          title="Context-aware character fixes + spell checking"
        >
          <SpellCheck2 size={14} /> Auto-correct
        </button>
      </div>
    </div>
  );
}
