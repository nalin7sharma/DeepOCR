import { useEffect, useMemo, useRef, useState } from "react";
import { Camera, ClipboardPaste, Upload, X } from "lucide-react";

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
export const ACCEPTED_TYPES = ".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp";
const ACCEPTED = /\.(pdf|png|jpe?g|bmp|tiff?|webp)$/i;

export function validateFile(file: File): string {
  if (file.size > MAX_UPLOAD_BYTES) return "This file is larger than 100 MB. Try a smaller scan or fewer pages.";
  if (file.size === 0) return "This file is empty.";
  if (!ACCEPTED.test(file.name) && !file.type.startsWith("image/") && file.type !== "application/pdf") {
    return "Unsupported file type. Use a PDF, PNG, JPG, TIFF or WEBP.";
  }
  return "";
}

export function useFilePreview(file: File | null) {
  const url = useMemo(() => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);
  return url;
}

interface Props {
  file: File | null;
  onFile: (file: File | null) => void;
  compact?: boolean;
}

export function Dropzone({ file, onFile, compact }: Props) {
  const [dragging, setDragging] = useState(false);
  const cameraRef = useRef<HTMLInputElement>(null);
  const preview = useFilePreview(file);

  // Paste a screenshot anywhere on the page with Ctrl+V.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const item = [...(event.clipboardData?.items ?? [])].find((entry) => entry.kind === "file");
      const pasted = item?.getAsFile();
      if (pasted) {
        const name = pasted.name && pasted.name !== "image.png" ? pasted.name : `pasted-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.png`;
        onFile(new File([pasted], name, { type: pasted.type }));
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile]);

  if (file) {
    const isPdf = file.name.toLowerCase().endsWith(".pdf");
    return (
      <div className="drop has-file">
        <div className="file-thumb">{preview ? <img src={preview} alt="" /> : isPdf ? "PDF" : "FILE"}</div>
        <div className="file-meta">
          <b title={file.name}>{file.name}</b>
          <span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready</span>
        </div>
        <button type="button" className="icon-btn sm" aria-label="Remove file" onClick={() => onFile(null)}><X size={15} /></button>
      </div>
    );
  }

  return (
    <label
      className={`drop ${dragging ? "drag" : ""}`}
      style={compact ? { padding: 16 } : undefined}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        onFile(event.dataTransfer.files[0] ?? null);
      }}
    >
      <input
        type="file"
        accept={ACCEPTED_TYPES}
        aria-label="Choose document"
        onChange={(event) => {
          onFile(event.target.files?.[0] ?? null);
          event.target.value = "";
        }}
      />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(event) => {
          onFile(event.target.files?.[0] ?? null);
          event.target.value = "";
        }}
      />
      {!compact && <span className="drop-icon"><Upload size={20} /></span>}
      <b>{dragging ? "Release to add the file" : <>Drop a file here, or <span className="browse">browse</span></>}</b>
      <span className="muted">PDF, PNG, JPG, TIFF or WEBP · up to 100 MB</span>
      {!compact && (
        <span className="drop-actions">
          <span className="btn sm" aria-hidden="true"><ClipboardPaste size={14} />Paste <span className="kbd light">Ctrl V</span></span>
          <button
            type="button"
            className="btn sm"
            onClick={(event) => {
              event.preventDefault();
              cameraRef.current?.click();
            }}
          >
            <Camera size={14} />Camera
          </button>
        </span>
      )}
    </label>
  );
}
