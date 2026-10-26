import { useEffect, useRef, useState } from "react";
import { fetchEngines, fetchMetrics, fetchSamples } from "./api";
import type { EnginesResponse, MetricsResponse, SampleDoc } from "./types";

/** Poll a JSON endpoint; keeps the last good value when a poll fails. */
function usePolling<T>(loader: (signal: AbortSignal) => Promise<T>, intervalMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string>("");
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const controller = new AbortController();
    const run = async () => {
      try {
        const value = await loader(controller.signal);
        if (alive) {
          setData(value);
          setError("");
        }
      } catch (caught) {
        if (alive && !(caught instanceof DOMException && caught.name === "AbortError")) {
          setError(caught instanceof Error ? caught.message : "Request failed");
        }
      }
      if (alive && intervalMs > 0) timer = window.setTimeout(run, intervalMs);
    };
    run();
    return () => {
      alive = false;
      controller.abort();
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, tick]);
  return { data, error, refresh: () => setTick((value) => value + 1) };
}

export const useEngines = () => usePolling<EnginesResponse>(fetchEngines, 8000);
export const useMetrics = (fast: boolean) => usePolling<MetricsResponse>(fetchMetrics, fast ? 20000 : 60000);
export const useSamples = () => usePolling<SampleDoc[]>(fetchSamples, 0);

/** Animate a number from its previous value to `target`. */
export function useCountUp(target: number, duration = 900) {
  const [value, setValue] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    const start = performance.now();
    const initial = from.current;
    let frame = 0;
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const next = initial + (target - initial) * eased;
      setValue(next);
      if (progress < 1) frame = requestAnimationFrame(step);
      else from.current = target;
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);
  return value;
}

export function useHashRoute<T extends string>(routes: readonly T[], fallback: T): [T, (route: T) => void] {
  const read = () => {
    const hash = window.location.hash.replace(/^#\/?/, "") as T;
    return routes.includes(hash) ? hash : fallback;
  };
  const [route, setRoute] = useState<T>(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return [route, (next: T) => {
    window.location.hash = `/${next}`;
    setRoute(next);
  }];
}

export function useStoredState<T extends string>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return (window.localStorage.getItem(key) as T) || initial;
    } catch {
      return initial;
    }
  });
  return [value, (next: T) => {
    setValue(next);
    try {
      window.localStorage.setItem(key, next);
    } catch {
      // Storage can be unavailable (private mode); the preference just isn't remembered.
    }
  }];
}
