/** Animated aurora + grid + grain backdrop (purely decorative). */
export function Background() {
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="aurora a1" />
      <div className="aurora a2" />
      <div className="aurora a3" />
      <div className="grid-lines" />
      <div className="grain" />
    </div>
  );
}
