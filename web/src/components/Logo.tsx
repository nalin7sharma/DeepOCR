export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="deepocr-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366f1" />
          <stop offset="1" stopColor="#06b6d4" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="28" height="28" rx="8" fill="url(#deepocr-logo)" />
      <rect x="9" y="10" width="14" height="2.4" rx="1.2" fill="#fff" />
      <rect x="9" y="15" width="10" height="2.4" rx="1.2" fill="#fff" opacity="0.8" />
      <rect x="9" y="20" width="12" height="2.4" rx="1.2" fill="#fff" opacity="0.6" />
    </svg>
  );
}
