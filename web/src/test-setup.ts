import "@testing-library/jest-dom/vitest";

// jsdom lacks a few browser APIs used by the UI / framer-motion.
class IntersectionObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() { return []; }
}
Object.defineProperty(window, "IntersectionObserver", { configurable: true, value: IntersectionObserverStub });
Object.defineProperty(window, "scrollTo", { configurable: true, value: () => {} });
Object.defineProperty(Element.prototype, "scrollIntoView", { configurable: true, value: () => {} });
