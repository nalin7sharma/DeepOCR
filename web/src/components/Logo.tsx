/** Animated mark: a page with text lines being swept by a scan beam. */
export function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="logo-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#a78bfa" />
          <stop offset="0.55" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#a3e635" />
        </linearGradient>
      </defs>
      <rect x="3" y="3" width="26" height="26" rx="8" fill="url(#logo-grad)" />
      <rect x="4.5" y="4.5" width="23" height="23" rx="6.5" fill="#0b0b16" />
      <rect x="9" y="10" width="14" height="2.2" rx="1.1" fill="#ecebff" opacity="0.95" />
      <rect x="9" y="15" width="10" height="2.2" rx="1.1" fill="#ecebff" opacity="0.7" />
      <rect x="9" y="20" width="12" height="2.2" rx="1.1" fill="#ecebff" opacity="0.5" />
      <rect className="logo-beam" x="6" y="8" width="20" height="1.6" rx="0.8" fill="#22d3ee" />
    </svg>
  );
}
