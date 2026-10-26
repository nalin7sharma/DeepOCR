import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Camera, ClipboardPaste, FileText, ImageUp, X } from "lucide-react";

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
export const ACCEPTED_TYPES = ".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp";
const ACCEPTED = /\.(pdf|png|jpe?g|bmp|tiff?|webp)$/i;

export function validateFile(file: File): string {
  if (file.size > MAX_UPLOAD_BYTES) return "This file exceeds the 100 MB upload limit. Try a smaller scan.";
  if (file.size === 0) return "This file is empty.";
  if (!ACCEPTED.test(file.name) && !file.type.startsWith("image/") && file.type !== "application/pdf") {
    return "Unsupported file type. Use a PDF, PNG, JPG, BMP, TIFF or WEBP.";
  }
  return "";
}

interface Props {
  file: File | null;
  onFile: (file: File | null) => void;
  compact?: boolean;
}

export function Dropzone({ file, onFile, compact }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const preview = useMemo(() => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  // Paste an image from the clipboard anywhere on the page (Ctrl+V).
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const item = [...(event.clipboardData?.items ?? [])].find((entry) => entry.kind === "file");
      const pasted = item?.getAsFile();
      if (pasted) {
        const named = new File([pasted], pasted.name && pasted.name !== "image.png" ? pasted.name : `pasted-${Date.now()}.png`, { type: pasted.type });
        onFile(named);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile]);

  const isPdf = file?.name.toLowerCase().endsWith(".pdf");

  return (
    <motion.label
      layout
      className={`dropzone ${compact ? "compact" : ""} ${file ? "has-file" : ""} ${dragging ? "dragging" : ""}`}
      htmlFor="document-upload"
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
        id="document-upload"
        ref={inputRef}
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
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(event) => {
          onFile(event.target.files?.[0] ?? null);
          event.target.value = "";
        }}
      />
      <div className="dz-ring" aria-hidden="true" />
      {file ? (
        <div className="dz-file">
          <div className="dz-thumb">
            {preview ? <img src={preview} alt="Selected document preview" /> : <span className="dz-pdf"><FileText size={28} />{isPdf ? "PDF" : "FILE"}</span>}
            <span className="dz-scan" aria-hidden="true" />
          </div>
          <div className="dz-meta">
            <strong title={file.name}>{file.name}</strong>
            <span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready to read</span>
          </div>
          <button
            type="button"
            className="dz-clear"
            aria-label="Remove file"
            onClick={(event) => {
              event.preventDefault();
              onFile(null);
            }}
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        <div className="dz-empty">
          <motion.span className="dz-icon" animate={{ y: dragging ? -6 : [0, -4, 0] }} transition={{ repeat: dragging ? 0 : Infinity, duration: 2.6 }}>
            <ImageUp size={30} />
          </motion.span>
          <strong>{dragging ? "Release to drop it in" : "Drop a document, or click to browse"}</strong>
          <span>PDF · PNG · JPG · TIFF · WEBP — up to 100 MB, 100 pages</span>
          <div className="dz-actions">
            <span className="kbd-hint"><ClipboardPaste size={14} /> Paste with <kbd>Ctrl</kbd>+<kbd>V</kbd></span>
            <button
              type="button"
              className="ghost-button small"
              onClick={(event) => {
                event.preventDefault();
                cameraRef.current?.click();
              }}
            >
              <Camera size={14} /> Camera
            </button>
          </div>
        </div>
      )}
    </motion.label>
  );
}
