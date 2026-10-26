import { Segmented, Switch } from "./Ui";

export type Cleanup = "auto" | "scan" | "photo" | "off";

/** Image-cleanup presets mapped to the API's preprocessing steps. */
export const CLEANUP_STEPS: Record<Cleanup, string[]> = {
  auto: ["deskew", "contrast"],
  scan: ["deskew", "denoise", "contrast"],
  photo: ["deskew", "denoise", "contrast", "binarize"],
  off: [],
};

const HINTS: Record<Cleanup, string> = {
  auto: "Straightens tilted pages and boosts faint text.",
  scan: "For photocopies: also removes speckle noise.",
  photo: "For phone photos: also evens out shadows and lighting.",
  off: "Uses the image exactly as uploaded.",
};

export function cleanupFromSteps(steps: string[] | undefined): Cleanup {
  if (!steps) return "auto";
  const key = [...steps].sort().join(",");
  return (Object.keys(CLEANUP_STEPS) as Cleanup[]).find((preset) => [...CLEANUP_STEPS[preset]].sort().join(",") === key) ?? "auto";
}

interface Props {
  cleanup: Cleanup;
  onCleanup: (value: Cleanup) => void;
  correct: boolean;
  onCorrect: (value: boolean) => void;
}

export function Options({ cleanup, onCleanup, correct, onCorrect }: Props) {
  return (
    <div className="opts">
      <div>
        <span className="opt-label" id="cleanup-label">Image cleanup</span>
        <Segmented<Cleanup>
          label="Image cleanup"
          value={cleanup}
          onChange={onCleanup}
          options={[
            { value: "auto", label: "Auto" },
            { value: "scan", label: "Scan" },
            { value: "photo", label: "Photo" },
            { value: "off", label: "Off" },
          ]}
        />
        <span className="hint">{HINTS[cleanup]}</span>
      </div>
      <div className="opt-switch">
        <Switch on={correct} onChange={onCorrect} label="Auto-correct" />
        <span><b>Auto-correct</b><span className="hint">Fixes OCR typos; each change is shown and can be undone.</span></span>
      </div>
    </div>
  );
}
