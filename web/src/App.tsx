import { useEffect } from "react";
import { BookOpen, FlaskConical, GitCompareArrows, ScanText, Workflow } from "lucide-react";
import { Logo } from "./components/Logo";
import { ToastProvider } from "./components/Ui";
import { useEngines, useHashRoute, useMetrics, useSamples } from "./hooks";
import { Studio } from "./pages/Studio";
import { Compare } from "./pages/Compare";
import { Research } from "./pages/Research";
import { HowItWorks } from "./pages/HowItWorks";

const ROUTES = ["studio", "compare", "research", "how-it-works"] as const;
type Route = (typeof ROUTES)[number];

const NAV: { id: Route; label: string; icon: typeof ScanText }[] = [
  { id: "studio", label: "Studio", icon: ScanText },
  { id: "compare", label: "Compare", icon: GitCompareArrows },
  { id: "research", label: "Research", icon: FlaskConical },
  { id: "how-it-works", label: "How it works", icon: Workflow },
];

function App() {
  const [route, setRoute] = useHashRoute(ROUTES, "studio");
  const engines = useEngines();
  const metrics = useMetrics(false);
  const samples = useSamples();

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [route]);

  const list = engines.data?.engines ?? [];
  const ready = list.filter((engine) => engine.available).length;
  const statusClass = engines.error ? "off" : ready ? "" : "warn";
  const statusText = engines.error ? "API offline" : list.length ? `${ready} engines ready` : "Connecting…";

  return (
    <ToastProvider>
      <div className="app">
        <header className="topbar">
          <a className="brand" href="#/studio" onClick={() => setRoute("studio")} aria-label="DeepOCR home">
            <Logo />
            <span>Deep<b>OCR</b></span>
          </a>
          <nav className="tabs" aria-label="Sections">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`tab ${route === id ? "on" : ""}`}
                aria-current={route === id ? "page" : undefined}
                onClick={() => setRoute(id)}
                title={label}
              >
                <Icon size={16} aria-hidden="true" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
          <div className="top-right">
            <span className="status" title={engines.error || "Recognition engines"}>
              <i className={`dot ${statusClass}`} />
              <span className="txt">{statusText}</span>
            </span>
            <a className="icon-btn" href="/api/docs" target="_blank" rel="noreferrer" aria-label="API documentation" title="API documentation">
              <BookOpen size={16} />
            </a>
          </div>
        </header>

        <main className="main">
          {route === "studio" && <Studio engines={engines.data} samples={samples.data ?? []} onEnginesChanged={engines.refresh} />}
          {route === "compare" && <Compare engines={engines.data} samples={samples.data ?? []} />}
          {route === "research" && <Research metrics={metrics.data} error={metrics.error} />}
          {route === "how-it-works" && <HowItWorks onTry={() => setRoute("studio")} />}
        </main>

        <footer className="footer">
          <span>DeepOCR · deep learning OCR for handwritten and complex documents</span>
          <span>Files are processed on this machine and never stored. Review automated output before relying on it.</span>
        </footer>
      </div>
    </ToastProvider>
  );
}

export default App;
