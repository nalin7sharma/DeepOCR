import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookOpenText, FlaskConical, GitCompareArrows, Moon, ScanText, Sun, Workflow } from "lucide-react";
import { Background } from "./components/Background";
import { Logo } from "./components/Logo";
import { useEngines, useHashRoute, useMetrics, useSamples, useStoredState } from "./hooks";
import { Studio } from "./pages/Studio";
import { Compare } from "./pages/Compare";
import { Research } from "./pages/Research";
import { Pipeline } from "./pages/Pipeline";

const ROUTES = ["studio", "compare", "research", "pipeline"] as const;
type Route = (typeof ROUTES)[number];

const NAV: { id: Route; label: string; icon: typeof ScanText }[] = [
  { id: "studio", label: "Studio", icon: ScanText },
  { id: "compare", label: "Compare", icon: GitCompareArrows },
  { id: "research", label: "Research", icon: FlaskConical },
  { id: "pipeline", label: "Pipeline", icon: Workflow },
];

function App() {
  const [route, setRoute] = useHashRoute(ROUTES, "studio");
  const [theme, setTheme] = useStoredState<"dark" | "light">("deepocr-theme", "dark");
  const engines = useEngines();
  const training = Object.values(engines.data?.engines ?? []).some((engine) => engine.info?.training);
  const metrics = useMetrics(training);
  const samples = useSamples();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [route]);

  const ready = engines.data?.engines.filter((engine) => engine.available).length ?? 0;
  const total = engines.data?.engines.length ?? 0;

  return (
    <div className="app">
      <Background />
      <header className="nav">
        <a className="nav-brand" href="#/studio" aria-label="DeepOCR home" onClick={() => setRoute("studio")}>
          <Logo />
          <span className="brand-word">Deep<span>OCR</span></span>
        </a>
        <nav className="nav-tabs" aria-label="Sections">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-tab ${route === id ? "active" : ""}`}
              aria-current={route === id ? "page" : undefined}
              onClick={() => setRoute(id)}
            >
              {route === id && <motion.span layoutId="nav-pill" className="nav-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
              <Icon size={16} aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="nav-right">
          <span className={`status-chip ${engines.error ? "offline" : ready ? "online" : "pending"}`} title={engines.error || "OCR engines"}>
            <i />
            {engines.error ? "API offline" : total ? `${ready}/${total} engines` : "Connecting…"}
            {training && <b className="training-dot">training</b>}
          </span>
          <a className="icon-button" href="/api/docs" target="_blank" rel="noreferrer" aria-label="API documentation" title="API docs">
            <BookOpenText size={17} />
          </a>
          <button
            type="button"
            className="icon-button"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>
      </header>

      <main className="main">
        <AnimatePresence mode="wait">
          <motion.div
            key={route}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            {route === "studio" && <Studio engines={engines.data} samples={samples.data ?? []} onEnginesChanged={engines.refresh} />}
            {route === "compare" && <Compare engines={engines.data} samples={samples.data ?? []} />}
            {route === "research" && <Research metrics={metrics.data} engines={engines.data} error={metrics.error} />}
            {route === "pipeline" && <Pipeline onTry={() => setRoute("studio")} />}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="footer">
        <span>DeepOCR · Deep Learning-Based Intelligent OCR for Handwritten & Complex Documents</span>
        <span>Files are processed in memory and never stored. Review automated output before relying on it.</span>
      </footer>
    </div>
  );
}

export default App;
