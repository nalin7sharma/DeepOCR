/**
 * Build presentation/DeepOCR_Presentation.pptx (15 slides) from the project's results.
 *
 *   cd presentation/src && npm install && node build_deck.js
 *
 * Then run finalize_deck.ps1 (Microsoft PowerPoint) to add Morph transitions and export
 * a PDF and slide images. Shapes that should glide between slides share a name that
 * starts with "!!", which makes PowerPoint's Morph match them across slides.
 */
const fs = require("fs");
const path = require("path");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const pptxgen = require("pptxgenjs");
const lu = require("react-icons/lu");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "presentation", "DeepOCR_Presentation.pptx");
const FIG = path.join(ROOT, "report", "figures");
const read = (rel) => JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8"));
const summary = read("results/evaluation/summary.json").models;
const ensemble = read("results/evaluation/ensemble.json");
const robustness = read("results/evaluation/robustness.json");
const hist = {
  cnn: read("results/training/cnn_bilstm_history.json"),
  crnn: read("results/training/crnn_history.json"),
  ft: read("results/training/trocr_ft_history.json"),
};

const THEME = {
  name: "DeepOCR",
  headFontFace: "Segoe UI Semibold",
  bodyFontFace: "Segoe UI",
  colors: {
    dk1: "101828", lt1: "FFFFFF", dk2: "1E1B4B", lt2: "F4F5F8",
    accent1: "4F46E5", accent2: "F79009", accent3: "16B364", accent4: "7A5AF8", accent5: "0BA5EC", accent6: "667085",
    hlink: "4F46E5", folHlink: "7A5AF8",
  },
};
const HEX = THEME.colors;
const SOFT_ON_DARK = "C7D2FE"; // secondary text on the dark slides
const MODEL = { ensemble: HEX.accent1, trocr_ft: HEX.accent2, trocr: HEX.accent3, crnn: HEX.accent4, cnn_bilstm: HEX.accent5, paddle: HEX.accent6 };

const TEAM = [
  ["Nalin Sharma", "CSJMA23001390292"],
  ["Amritansh Singh Gaur", "CSJMA23001390309"],
  ["Vineet Kumar Singh", "CSJMA23001390311"],
  ["Rajyavardhan Radhey", "CSJMA23001390322"],
];

const W = 13.333;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "DeepOCR: Deep Learning-Based Intelligent OCR System";
pres.subject = "Project presentation";
const C = pres.SchemeColor;

/* ------------------------------------------------------------------ assets */
const iconCache = new Map();
async function icon(name, color) {
  const key = `${name}-${color}`;
  if (!iconCache.has(key)) {
    const Icon = lu[name];
    if (!Icon) throw new Error(`Unknown icon ${name}`);
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { size: 256, color: `#${color}`, strokeWidth: 2 }));
    const png = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
    iconCache.set(key, `image/png;base64,${png.toString("base64")}`);
  }
  return iconCache.get(key);
}

async function fit(file, maxW, maxH) {
  const meta = await sharp(file).metadata();
  const scale = Math.min(maxW / meta.width, maxH / meta.height);
  return { w: meta.width * scale, h: meta.height * scale };
}

/* ------------------------------------------------------------------ layouts */
const footer = (color) => ({ text: { text: "DeepOCR · Intelligent OCR for handwritten and complex documents", options: { x: 0.6, y: 6.98, w: 7, h: 0.3, fontSize: 10, color, margin: 0 } } });
const titlePlaceholder = (color) => ({
  placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.42, w: 10.4, h: 0.85, fontSize: 30, color, align: "left", valign: "middle", margin: 0 }, text: "" },
});
pres.defineSlideMaster({
  title: "Content light",
  background: { color: C.background1 },
  objects: [titlePlaceholder(C.text1), footer(C.accent6)],
  slideNumber: { x: 12.13, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "Content dark",
  background: { color: C.text2 },
  objects: [titlePlaceholder(C.background1), footer(SOFT_ON_DARK)],
  slideNumber: { x: 12.13, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: SOFT_ON_DARK, align: "right" },
});
pres.defineSlideMaster({
  title: "Cover",
  background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.05, w: 7.2, h: 1.25, fontSize: 64, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});

/* ------------------------------------------------------------------ helpers */
// Keep hyphenated names on one line: "CNN-BiLSTM" must not break at its hyphen.
const NB = (value) => value.replace(/CNN-BiLSTM|cross-validation|fine-tuned|fine-tuning/g, (m) => m.replace(/-/g, "‑"));
const nb = (value) => (typeof value === "string" ? NB(value) : value.map((run) => ({ ...run, text: NB(run.text) })));
const text = (slide, value, opts) => slide.addText(nb(value), { isTextBox: true, margin: 0, color: C.text1, fontSize: 16, valign: "top", ...opts });

/** The recurring "lens": one circle per slide, named alike so Morph glides it between slides. */
function lens(slide, x, y, d, { dark = false, transparency } = {}) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: d, h: d, objectName: "!!lens",
    fill: { color: C.accent1, transparency: transparency ?? (dark ? 0 : 90) },
    line: { type: "none" },
  });
}

async function iconCircle(slide, name, x, y, d, { fill, iconColor, objectName, line } = {}) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: d, h: d, objectName,
    fill: { color: fill ?? C.background1 },
    line: line ?? { type: "none" },
  });
  const pad = d * 0.24;
  slide.addImage({ data: await icon(name, iconColor ?? HEX.accent1), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, altText: "" });
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12,
    fill: { color: C.background2, ...(opts.fill ?? {}) },
    line: opts.line ?? { type: "none" },
    objectName: opts.objectName,
  });
}

const shadow = () => ({ type: "outer", blur: 10, offset: 3, angle: 90, color: "101828", opacity: 0.16 });

const chartText = {
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.accent6, valAxisLabelColor: HEX.accent6, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  legendFontSize: 11, legendColor: "475467", titleFontSize: 13, titleColor: HEX.dk1, dataLabelFontSize: 10, dataLabelColor: "344054",
  valGridLine: { color: "E4E7EC", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
};

const f1 = (v) => v.toFixed(1);
const avgMs = (id) => Math.round((summary[id].iam.ms_per_line + summary[id].synthetic.ms_per_line) / 2);
const f2 = (v) => v.toFixed(2);

/* ================================================================== slides */
async function build() {
  /* ---------------------------------------------------------------- 1 cover */
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Cover", sectionTitle: "Introduction" });
  lens(s, 8.25, 0.85, 5.8, { dark: true });
  // The DeepOCR logo mark: three text lines inside the lens.
  [[2.5, 0], [1.75, 15], [2.1, 30]].forEach(([w, t], i) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 11.15 - 1.25, y: 3.18 + i * 0.5, w, h: 0.3, rectRadius: 0.15, objectName: `!!mark${i + 1}`,
    fill: { color: C.background1, transparency: t }, line: { type: "none" },
  }));
  text(s, "PROJECT PRESENTATION  ·  OCTOBER 2026", { x: 0.7, y: 1.45, w: 7.2, h: 0.4, fontSize: 14, bold: true, charSpacing: 4, color: SOFT_ON_DARK });
  s.addText("DeepOCR", { placeholder: "title" });
  text(s, "Deep Learning-Based Intelligent OCR System for Handwritten and Complex Document Recognition", { x: 0.7, y: 3.35, w: 7.2, h: 1.0, fontSize: 22, color: C.background1 });
  text(s, "PRESENTED BY", { x: 0.7, y: 4.75, w: 7.2, h: 0.3, fontSize: 12, bold: true, charSpacing: 3, color: SOFT_ON_DARK });
  TEAM.forEach(([name, roll], i) => {
    const x = 0.7 + (i % 2) * 3.7;
    const y = 5.2 + Math.floor(i / 2) * 0.82;
    text(s, name, { x, y, w: 3.5, h: 0.36, fontSize: 16, bold: true, color: C.background1 });
    text(s, roll, { x, y: y + 0.36, w: 3.5, h: 0.3, fontSize: 12, color: SOFT_ON_DARK });
  });
  s.addNotes("DeepOCR is an end-to-end OCR system for handwritten and complex documents. It cleans the page, finds every text line, reads it with deep-learning models, rebuilds the layout and lets a person review the result. Its key idea is that five different models vote on every word.");

  /* ---------------------------------------------------------------- 2 problem */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Introduction" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("Why OCR still fails on real documents", { placeholder: "title" });
  text(s, "Clean printed pages are easy. Real documents are not.", { x: 0.6, y: 1.32, w: 9, h: 0.4, fontSize: 18, color: C.accent6 });
  const problems = [
    ["LuPenLine", "Handwriting", "Every writer is different; letters touch and slant."],
    ["LuImageOff", "Poor scans", "Noise, blur, shadows, low contrast and faint ink."],
    ["LuLayoutGrid", "Skew and layout", "Tilted pages, tables, forms, checkboxes, columns."],
    ["LuTriangleAlert", "No sense of doubt", "Tools never say which words may be wrong."],
  ];
  for (const [i, [ic, title, desc]] of problems.entries()) {
    const x = 0.6 + (i % 2) * 3.8;
    const y = 2.05 + Math.floor(i / 2) * 2.45;
    card(s, x, y, 3.5, 2.2);
    await iconCircle(s, ic, x + 0.3, y + 0.3, 0.62);
    text(s, title, { x: x + 0.3, y: y + 1.08, w: 2.9, h: 0.4, fontSize: 18, fontFace: THEME.headFontFace });
    text(s, desc, { x: x + 0.3, y: y + 1.5, w: 2.95, h: 0.6, fontSize: 14, color: C.accent6 });
  }
  const scan = path.join(ROOT, "samples", "noisy_scan.jpg");
  const scanSize = await fit(scan, 4.4, 4.2);
  s.addImage({ path: scan, x: 8.3 + (4.4 - scanSize.w) / 2, y: 2.05, ...scanSize, shadow: shadow(), altText: "A noisy, skewed scanned notice" });
  text(s, "A test input: 4° skew, noise and uneven lighting", { x: 8.3, y: 2.15 + scanSize.h, w: 4.4, h: 0.35, fontSize: 11, color: C.accent6, align: "center" });
  s.addNotes("Commercial OCR handles clean print well. It struggles with handwriting, poor scans, tilted pages and structured layouts such as forms and tables. It also gives no hint about which words might be wrong, so users must re-read everything.");

  /* ---------------------------------------------------------------- 3 objectives */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Introduction" });
  lens(s, -2.6, 4.4, 3.0);
  s.addText("Objectives from the project proposal", { placeholder: "title" });
  const objectives = [
    "Extract text from document images", "Recognise handwritten and printed text",
    "Clean noisy, blurred and skewed images", "Detect text regions in complex documents",
    "Analyse the document layout", "Compare deep-learning recognition models",
    "Evaluate with standard metrics (CER, WER)", "Build a user-friendly interface",
  ];
  objectives.forEach((label, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2;
    const y = 1.75 + row * 1.2;
    card(s, x, y, 5.9, 0.95);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.19, w: 0.57, h: 0.57, fill: { color: C.accent1 }, line: { type: "none" } });
    text(s, String(i + 1), { x: x + 0.2, y: y + 0.19, w: 0.57, h: 0.57, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, label, { x: x + 0.95, y: y + 0.1, w: 4.8, h: 0.75, fontSize: 17, valign: "middle" });
  });
  s.addNotes("These are the eight objectives from the project proposal. The rest of the presentation shows how each one was met, from preprocessing and detection to the model comparison and the user interface.");

  /* ---------------------------------------------------------------- 4 pipeline */
  pres.addSection({ title: "System" });
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "System" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("System overview: an eight-stage pipeline", { placeholder: "title" });
  text(s, "From an uploaded file to reviewed, structured text", { x: 0.6, y: 1.32, w: 9, h: 0.4, fontSize: 18, color: C.accent6 });
  const stages = [
    ["LuFileInput", "Input", "PDF or image, up to 100 pages"],
    ["LuWandSparkles", "Clean up", "Deskew, denoise, CLAHE contrast"],
    ["LuScanLine", "Detect lines", "DBNet finds every text line"],
    ["LuBrainCircuit", "Recognise", "CNN-BiLSTM, CRNN or TrOCR"],
    ["LuVote", "Vote", "Five models vote on each word"],
    ["LuLayers", "Layout", "Headings, tables, checkboxes"],
    ["LuSpellCheck2", "Correct", "Fix 0/O, 1/l and spelling"],
    ["LuListChecks", "Review", "Overlays, review queue, export"],
  ];
  const step = 10.6 / 7;
  s.addShape(pres.shapes.LINE, { x: 1.15, y: 2.97, w: 10.6, h: 0, line: { color: "D0D5DD", width: 2 } });
  for (const [i, [ic, label, detail]] of stages.entries()) {
    const cx = 1.15 + i * step;
    const vote = i === 4;
    await iconCircle(s, ic, cx - 0.47, 2.5, 0.94, {
      fill: vote ? C.accent1 : C.background1, iconColor: vote ? HEX.lt1 : HEX.accent1,
      line: vote ? { type: "none" } : { color: C.accent1, width: 1.5 }, objectName: `!!stage${i + 1}`,
    });
    text(s, label, { x: cx - 0.75, y: 3.65, w: 1.5, h: 0.4, fontSize: 15, align: "center", fontFace: THEME.headFontFace, color: vote ? C.accent1 : C.text1 });
    text(s, detail, { x: cx - 0.74, y: 4.07, w: 1.48, h: 0.75, fontSize: 12, align: "center", color: C.accent6 });
  }
  text(s, "Only the Consensus engine runs the vote; single models go straight from recognition to layout.", { x: 0.6, y: 5.05, w: 12.1, h: 0.35, fontSize: 12, italic: true, color: C.accent6, align: "center" });
  card(s, 0.6, 5.65, 12.1, 0.85);
  text(s, [
    { text: "Built with  ", options: { color: C.accent6 } },
    { text: "Python · PyTorch · OpenCV · Hugging Face Transformers · PaddleOCR · FastAPI · React + TypeScript", options: { color: C.text1 } },
  ], { x: 0.9, y: 5.65, w: 11.5, h: 0.85, fontSize: 15, valign: "middle", align: "center" });
  s.addNotes("A document passes through eight stages. The page is cleaned, DBNet finds every text line, and a recognition model reads each line. With the consensus engine, all five models read every line and vote word by word. Then the layout is rebuilt, typical OCR errors are corrected, and the result is shown for review.");

  /* ---------------------------------------------------------------- 5 data */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "System" });
  lens(s, 12.95, 2.3, 3.0);
  s.addText("Data: real handwriting plus synthetic print", { placeholder: "title" });
  const datasets = [
    ["10,373", "IAM handwriting lines · 657 writers", [["Train", 6482], ["Val", 976], ["Test", 2915]]],
    ["6,500", "Synthetic printed and script lines · 32 fonts", [["Train", 5005], ["Val", 520], ["Test", 975]]],
  ];
  datasets.forEach(([big, label, split], i) => {
    const y = 1.6 + i * 2.6;
    card(s, 0.6, y, 5.6, 2.35);
    text(s, big, { x: 0.9, y: y + 0.25, w: 5, h: 0.75, fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
    text(s, label, { x: 0.9, y: y + 1.0, w: 5, h: 0.4, fontSize: 15 });
    const total = split.reduce((a, [, n]) => a + n, 0);
    let x = 0.9;
    const colors = [C.accent1, C.accent5, C.accent2];
    split.forEach(([name, n], k) => {
      const w = Math.max(0.42, (n / total) * 5.0);
      s.addShape(pres.shapes.RECTANGLE, { x, y: y + 1.58, w: w - 0.04, h: 0.22, fill: { color: colors[k] }, line: { type: "none" } });
      x += w;
    });
    text(s, split.map(([name, n]) => `${name} ${n.toLocaleString("en-US")}`).join("   ·   "), { x: 0.9, y: y + 1.88, w: 5, h: 0.32, fontSize: 12, color: C.accent6 });
  });
  const dataFig = path.join(FIG, "fig_datasets.png");
  const dataSize = await fit(dataFig, 6.2, 3.3);
  s.addImage({ path: dataFig, x: 6.55, y: 1.6, ...dataSize, altText: "Example IAM and synthetic test lines" });
  card(s, 6.55, 1.75 + dataSize.h, 6.2, 6.55 - (1.75 + dataSize.h));
  text(s, [
    { text: "Writer-independent split: ", options: { bold: true } },
    { text: "test writers never appear in training.", options: { breakLine: true } },
    { text: "Augmentation: ", options: { bold: true } },
    { text: "rotation, scaling, noise, blur, brightness, perspective." },
  ], { x: 6.8, y: 1.75 + dataSize.h, w: 5.75, h: 6.55 - (1.75 + dataSize.h), fontSize: 14, valign: "middle", paraSpaceAfter: 6 });
  s.addNotes("The models learn from the IAM handwriting database, using its official split so that test writers are never seen in training, plus a synthetic dataset of printed and script-font lines created for this project with realistic degradations. Training images are augmented on the fly.");

  /* ---------------------------------------------------------------- 6 preprocessing */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "System" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("Cleaning the page before reading it", { placeholder: "title" });
  const pre = path.join(FIG, "fig_preprocessing.png");
  const preSize = await fit(pre, 7.5, 3.6);
  s.addImage({ path: pre, x: 0.6, y: 1.5, ...preSize, altText: "Noisy scan before and after preprocessing" });
  card(s, 0.6, 5.35, 7.5, 1.2, { fill: { color: C.accent1, transparency: 90 } });
  text(s, "65.5% → 18.9%", { x: 0.9, y: 5.35, w: 3.4, h: 1.2, fontSize: 30, bold: true, color: C.accent1, valign: "middle", fontFace: THEME.headFontFace });
  text(s, "CER of the CNN-BiLSTM on lines tilted by 8°, without and with our deskew step", { x: 4.3, y: 5.35, w: 3.6, h: 1.2, fontSize: 14, valign: "middle" });
  const steps = [
    ["LuRotateCw", "Deskew", "Projection profile finds the tilt (±12°) and straightens the page"],
    ["LuSparkles", "Denoise", "Gaussian and median filtering remove speckle"],
    ["LuEye", "Contrast", "CLAHE lifts faint ink without boosting noise"],
    ["LuImage", "Binarise", "Adaptive threshold evens out phone-photo lighting"],
  ];
  for (const [i, [ic, title, desc]] of steps.entries()) {
    const y = 1.55 + i * 1.25;
    await iconCircle(s, ic, 8.5, y, 0.62, { fill: C.background2 });
    text(s, title, { x: 9.3, y: y - 0.02, w: 3.4, h: 0.38, fontSize: 17, fontFace: THEME.headFontFace });
    text(s, desc, { x: 9.3, y: y + 0.36, w: 3.45, h: 0.7, fontSize: 13, color: C.accent6 });
  }
  s.addNotes("Before recognition the page is cleaned. Deskewing uses a projection profile: the angle that gives the sharpest row profile aligns the text. Here a 4.2-degree tilt is found and corrected. The robustness study shows why this matters: at an 8-degree tilt the baseline's error rises to 65.5 percent, and deskewing brings it back to 18.9 percent.");

  /* ---------------------------------------------------------------- 7 models */
  pres.addSection({ title: "Models" });
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Models" });
  lens(s, -2.6, 4.2, 3.0);
  s.addText("Three models, from baseline to Transformer", { placeholder: "title" });
  const models = [
    ["cnn", "LuCpu", HEX.accent5, "BASELINE", "CNN-BiLSTM", "4 conv blocks + 2-layer BiLSTM + CTC", [["Parameters", "2.39 M"], ["Input", "32 × 512 grey"], ["Training", "From scratch, 124 min on CPU"], ["Speed", `${avgMs("cnn_bilstm")} ms per line`]]],
    ["crnn", "LuLayers", HEX.accent4, "MODEL 2", "CRNN", "7-layer VGG CNN + BiLSTM + CTC (Shi et al.)", [["Parameters", "2.78 M"], ["Input", "32 × 512 grey"], ["Training", "From scratch, 130 min on CPU"], ["Speed", `${avgMs("crnn")} ms per line`]]],
    ["trocr", "LuBrainCircuit", HEX.accent2, "ADVANCED", "TrOCR (fine-tuned)", "Vision Transformer encoder + text decoder", [["Parameters", "61.6 M"], ["Input", "384 × 384 RGB"], ["Training", "Pretrained, fine-tuned 44 min"], ["Speed", `${avgMs("trocr_ft")} ms per line`]]],
  ];
  for (const [i, [id, ic, color, role, name, arch, facts]] of models.entries()) {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.6, 3.85, 4.85, { objectName: `!!card-${id}` });
    await iconCircle(s, ic, x + 0.3, 1.9, 0.66, { fill: color, iconColor: HEX.lt1 });
    text(s, role, { x: x + 1.15, y: 1.95, w: 2.5, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: C.accent6 });
    text(s, name, { x: x + 1.15, y: 2.22, w: 2.6, h: 0.42, fontSize: 20, fontFace: THEME.headFontFace });
    text(s, arch, { x: x + 0.3, y: 2.8, w: 3.3, h: 0.65, fontSize: 14, color: C.text1 });
    facts.forEach(([label, value], k) => {
      const y = 3.6 + k * 0.68;
      text(s, label, { x: x + 0.3, y, w: 3.2, h: 0.25, fontSize: 11, color: C.accent6 });
      text(s, value, { x: x + 0.3, y: y + 0.25, w: 3.3, h: 0.35, fontSize: 15 });
    });
  }
  text(s, "Also compared: PaddleOCR (industrial PP-OCR) as a reference, and the Consensus vote of all five", { x: 0.6, y: 6.52, w: 12.1, h: 0.32, fontSize: 12, color: C.accent6 });
  s.addNotes("Three model families from the proposal. The baseline CNN-BiLSTM and the CRNN were built and trained from scratch on a laptop CPU in about two hours each. The advanced model is TrOCR, a Vision Transformer with a text decoder, used through transfer learning: we fine-tuned the pretrained model on our data in 44 minutes.");

  /* ---------------------------------------------------------------- 8 training */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Models" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("Training and fine-tuning on a laptop CPU", { placeholder: "title" });
  card(s, 0.6, 1.45, 3.05, 0.55, { objectName: "!!card-cnn" });
  text(s, `CNN-BiLSTM · best val CER ${f1(hist.cnn.best_val_cer * 100)}%`, { x: 0.6, y: 1.45, w: 3.05, h: 0.55, fontSize: 13, align: "center", valign: "middle" });
  card(s, 3.85, 1.45, 3.05, 0.55, { objectName: "!!card-crnn" });
  text(s, `CRNN · best val CER ${f1(hist.crnn.best_val_cer * 100)}%`, { x: 3.85, y: 1.45, w: 3.05, h: 0.55, fontSize: 13, align: "center", valign: "middle" });
  card(s, 7.4, 1.45, 5.3, 0.55, { objectName: "!!card-trocr" });
  text(s, "TrOCR fine-tuning · 300 steps · 44 min", { x: 7.4, y: 1.45, w: 5.3, h: 0.55, fontSize: 13, align: "center", valign: "middle" });
  const epochs = hist.cnn.history.val_cer.map((_, i) => String(i + 1));
  s.addChart(pres.charts.LINE, [
    { name: "CNN-BiLSTM", labels: epochs, values: hist.cnn.history.val_cer.map((v) => +(v * 100).toFixed(1)) },
    { name: "CRNN", labels: epochs, values: hist.crnn.history.val_cer.map((v) => +(v * 100).toFixed(1)) },
  ], {
    x: 0.6, y: 2.2, w: 6.3, h: 4.35, ...chartText,
    chartColors: [HEX.accent5, HEX.accent4], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showTitle: true, title: "Validation CER (%) per epoch", showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, showCatAxisTitle: true, catAxisTitle: "Epoch", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.accent6,
  });
  const ft = hist.ft.history;
  s.addChart(pres.charts.LINE, [
    { name: "Printed & degraded", labels: ft.step.map(String), values: ft.val_cer_synthetic.map((v) => +v.toFixed(1)) },
    { name: "Handwriting", labels: ft.step.map(String), values: ft.val_cer_iam.map((v) => +v.toFixed(1)) },
  ], {
    x: 7.4, y: 2.2, w: 5.3, h: 2.95, ...chartText,
    chartColors: [HEX.accent2, HEX.accent3], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showTitle: true, title: "TrOCR validation CER (%) by step", showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 25, valAxisMajorUnit: 5,
  });
  text(s, `${f2(summary.trocr.synthetic.cer)}% → ${f2(summary.trocr_ft.synthetic.cer)}%`, { x: 7.4, y: 5.3, w: 5.3, h: 0.6, fontSize: 28, bold: true, color: C.accent2, fontFace: THEME.headFontFace });
  text(s, [
    { text: "printed-text test CER, before → after fine-tuning", options: { breakLine: true } },
    { text: `handwriting unchanged: ${f2(summary.trocr.iam.cer)}% → ${f2(summary.trocr_ft.iam.cer)}%` },
  ], { x: 7.4, y: 5.9, w: 5.3, h: 0.6, fontSize: 13, color: C.accent6 });
  s.addNotes("Both CTC models converge within a few epochs; the best validation CERs are 11.8 and 12.2 percent on the mixed validation set. Fine-tuning TrOCR mainly fixed its weakness on printed text: test CER fell from 20.4 to 2.6 percent, while its handwriting accuracy stayed the same.");

  /* ---------------------------------------------------------------- 9 voting idea */
  pres.addSection({ title: "Consensus" });
  s = pres.addSlide({ masterName: "Content dark", sectionTitle: "Consensus" });
  lens(s, 11.3, -2.0, 3.4, { dark: true, transparency: 30 });
  s.addText("Key idea: let five models vote on every word", { placeholder: "title" });
  const ideas = [
    "Every model reads each text line",
    "The readings are aligned word by word",
    "Each word is decided by a weighted vote: model weight × confidence",
  ];
  ideas.forEach((idea, i) => {
    const y = 1.75 + i * 1.05;
    s.addShape(pres.shapes.OVAL, { x: 0.7, y, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { type: "none" } });
    text(s, String(i + 1), { x: 0.7, y, w: 0.6, h: 0.6, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, idea, { x: 1.55, y: y - 0.05, w: 4.6, h: 0.75, fontSize: 18, color: C.background1, valign: "middle" });
  });
  card(s, 0.7, 4.95, 5.45, 1.55, { fill: { color: C.background1, transparency: 90 } });
  await iconCircle(s, "LuTriangleAlert", 0.95, 5.2, 0.6, { fill: C.accent2, iconColor: HEX.lt1 });
  text(s, [
    { text: "Words without a majority are flagged", options: { bold: true, breakLine: true } },
    { text: "so the reviewer checks only the uncertain ones", options: { color: SOFT_ON_DARK } },
  ], { x: 1.75, y: 5.1, w: 4.25, h: 1.25, fontSize: 15, color: C.background1, valign: "middle" });
  card(s, 6.75, 1.6, 5.95, 4.9, { fill: { color: C.background1 }, objectName: "!!panel" });
  text(s, "Real example from the test page: “Tin ___ Alley”", { x: 7.05, y: 1.8, w: 5.4, h: 0.35, fontSize: 14, color: C.accent6 });
  const readings = [["trocr_ft", "TrOCR (fine-tuned)", 3, "fan"], ["crnn", "CRNN", 1, "fan"], ["trocr", "TrOCR (pretrained)", 3, "Pan"], ["paddle", "PaddleOCR", 2, "Pan"], ["cnn_bilstm", "CNN-BiLSTM", 2, "Pan"]];
  readings.forEach(([id, name, weight, word], i) => {
    const y = 2.3 + i * 0.47;
    s.addShape(pres.shapes.OVAL, { x: 7.05, y: y + 0.1, w: 0.17, h: 0.17, fill: { color: MODEL[id] }, line: { type: "none" } });
    text(s, `${name}  (weight ${weight})`, { x: 7.35, y, w: 3.4, h: 0.38, fontSize: 14, valign: "middle" });
    text(s, word, { x: 10.9, y, w: 1.5, h: 0.38, fontSize: 17, bold: true, valign: "middle", color: word === "Pan" ? C.accent1 : "D92D20" });
  });
  [["Pan", 66, C.accent1], ["fan", 34, "FDA29B"]].forEach(([word, share, color], i) => {
    const y = 4.85 + i * 0.5;
    text(s, word, { x: 7.05, y, w: 0.7, h: 0.38, fontSize: 15, bold: true, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.8, y: y + 0.06, w: (share / 100) * 3.8, h: 0.26, rectRadius: 0.13, fill: { color }, line: { type: "none" } });
    text(s, `${share}%`, { x: 7.9 + (share / 100) * 3.8, y, w: 0.8, h: 0.38, fontSize: 14, bold: true, valign: "middle" });
  });
  text(s, "Result: “Tin Pan Alley”, which is correct", { x: 7.05, y: 5.88, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: C.accent1 });
  s.addNotes("This is the main idea of the project. Different models make different mistakes. If several independent models read a word the same way, it is probably right. On this real line, two models read 'fan' and three read 'Pan'. Weighted by each model's reliability and confidence, 'Pan' wins with 66 percent of the vote, and it is correct. Words that win no majority are flagged for review.");

  /* ---------------------------------------------------------------- 10 results */
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Results" });
  lens(s, 12.95, 4.4, 3.0);
  s.addText("Results on 600 held-out lines per test set", { placeholder: "title" });
  const order = ["ensemble", "trocr_ft", "trocr", "crnn", "cnn_bilstm", "paddle"];
  const names = { ensemble: "Consensus vote", trocr_ft: "TrOCR (fine-tuned)", trocr: "TrOCR (pretrained)", crnn: "CRNN", cnn_bilstm: "CNN-BiLSTM", paddle: "PaddleOCR" };
  s.addChart(pres.charts.BAR, [
    { name: "Handwriting CER (%)", labels: order.map((m) => names[m]), values: order.map((m) => +summary[m].iam.cer.toFixed(1)) },
    { name: "Printed CER (%)", labels: order.map((m) => names[m]), values: order.map((m) => +summary[m].synthetic.cer.toFixed(1)) },
  ], {
    x: 0.6, y: 1.45, w: 7.9, h: 5.15, ...chartText, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 55,
    catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" },
    chartColors: [HEX.accent1, HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1,
    showTitle: true, title: "Character error rate by engine (lower is better)",
  });
  const stats = [
    ["!!stat-hw", `${f2(summary.ensemble.iam.cer)}%`, "CER on handwriting from unseen writers (Consensus)", C.accent1],
    ["!!stat-pr", `${f2(summary.ensemble.synthetic.cer)}%`, "CER on printed and degraded text (Consensus)", C.accent2],
    ["!!stat-sp", `${avgMs("cnn_bilstm")} ms`, "per line with CNN-BiLSTM, our fastest model", C.text1],
  ];
  stats.forEach(([name, big, label, color], i) => {
    const y = 1.6 + i * 1.7;
    text(s, big, { x: 8.95, y, w: 3.8, h: 0.8, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace, objectName: name });
    text(s, label, { x: 8.95, y: y + 0.8, w: 3.8, h: 0.6, fontSize: 14, color: C.accent6 });
  });
  s.addNotes("These are the results on 600 held-out lines per test set. Transformers dominate handwriting at about 7 percent CER, while our small CTC models are strong on print. The Consensus vote is the most accurate engine on both: 5.74 percent CER on handwriting from unseen writers and 2.35 percent on printed text. The trade-off is speed.");

  /* ---------------------------------------------------------------- 11 voting gain */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Results" });
  lens(s, -2.6, 3.9, 3.0);
  s.addText("Voting removes errors even the best model makes", { placeholder: "title" });
  const bestIam = Object.keys(ensemble.weights).map((m) => ensemble.sets.iam[m]).reduce((a, b) => (a.cer <= b.cer ? a : b));
  const bestPr = Object.keys(ensemble.weights).map((m) => ensemble.sets.synthetic[m]).reduce((a, b) => (a.cer <= b.cer ? a : b));
  const cv = (set) => ensemble.sets[set].ensemble_cv;
  const gain = (a, b) => `${Math.round(((a - b) / a) * 100)}%`;
  const big = [
    ["!!stat-hw", `${f2(cv("iam").cer)}%`, `handwriting CER: ${gain(bestIam.cer, cv("iam").cer)} fewer errors than the best single model (${f2(bestIam.cer)}%)`, C.accent1],
    ["!!stat-wer", `${f2(cv("synthetic").wer)}%`, `printed-text WER: ${gain(bestPr.wer, cv("synthetic").wer)} fewer word errors than the best single model (${f2(bestPr.wer)}%)`, C.accent2],
    ["!!stat-flag", "11 of 125", "words flagged as disputed on the handwritten demo page, where most errors are", C.text1],
  ];
  big.forEach(([name, value, label, color], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.5, 3.85, 2.15);
    text(s, value, { x: x + 0.3, y: 1.62, w: 3.3, h: 0.8, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace, objectName: name });
    text(s, label, { x: x + 0.3, y: 2.45, w: 3.3, h: 1.05, fontSize: 14, color: C.accent6 });
  });
  const oracle = (set) => ensemble.sets[set].oracle_best_line;
  s.addChart(pres.charts.BAR, [
    { name: "Best single model", labels: ["Handwriting CER", "Handwriting WER", "Print CER", "Print WER"], values: [bestIam.cer, bestIam.wer, bestPr.cer, bestPr.wer].map((v) => +v.toFixed(1)) },
    { name: "Consensus vote", labels: ["Handwriting CER", "Handwriting WER", "Print CER", "Print WER"], values: [cv("iam").cer, cv("iam").wer, cv("synthetic").cer, cv("synthetic").wer].map((v) => +v.toFixed(1)) },
    { name: "Oracle ceiling", labels: ["Handwriting CER", "Handwriting WER", "Print CER", "Print WER"], values: [oracle("iam").cer, oracle("iam").wer, oracle("synthetic").cer, oracle("synthetic").wer].map((v) => +v.toFixed(1)) },
  ], {
    x: 0.6, y: 3.9, w: 7.7, h: 2.75, ...chartText, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["98A2B3", HEX.accent1, "D0D5DD"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "r", valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0,
  });
  card(s, 8.7, 3.9, 4.0, 2.75);
  text(s, [
    { text: "Weights tuned by cross-validation", options: { bold: true, breakLine: true } },
    { text: "TrOCR-ft 3 · TrOCR 3 · PaddleOCR 2 · CNN-BiLSTM 2 · CRNN 1", options: { color: C.accent6, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Cost", options: { bold: true, breakLine: true } },
    { text: "About 1 s per line, because every model runs", options: { color: C.accent6 } },
  ], { x: 8.95, y: 4.05, w: 3.55, h: 2.45, fontSize: 14, valign: "middle" });
  s.addNotes("Compared with the best single model on the same lines, voting removes 15 percent of the errors on handwriting and 46 percent of the word errors on printed text. The oracle shows the best possible combination if we knew the answer. The weights were tuned with cross-validation, so the scores are not measured on the tuning data.");

  /* ---------------------------------------------------------------- 12 robustness */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Results" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("Robust to noise, and deskewing fixes rotation", { placeholder: "title" });
  const levels = robustness.conditions.find((c) => c.id === "rotation").levels.map((l) => `${l}°`);
  s.addChart(pres.charts.LINE, [
    { name: "CNN-BiLSTM, no deskew", labels: levels, values: robustness.models.cnn_bilstm.rotation },
    { name: "CNN-BiLSTM + our deskew", labels: levels, values: robustness.models.cnn_bilstm.rotation_deskew },
    { name: "PaddleOCR, no deskew", labels: levels, values: robustness.models.paddle.rotation },
    { name: "PaddleOCR + our deskew", labels: levels, values: robustness.models.paddle.rotation_deskew },
    { name: "TrOCR, no deskew", labels: levels, values: robustness.models.trocr.rotation },
  ], {
    x: 0.6, y: 1.45, w: 7.6, h: 5.15, ...chartText,
    chartColors: ["F04438", HEX.accent1, "FDA29B", "A5B4FC", HEX.accent3], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showTitle: true, title: "CER (%) as the text tilts (80 IAM test lines)", showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 90, valAxisMajorUnit: 30,
  });
  const cnnR = robustness.models.cnn_bilstm;
  const insights = [
    ["LuRotateCw", "Tilt breaks CTC models", `CNN-BiLSTM: ${f1(cnnR.rotation[0])}% → ${f1(cnnR.rotation.at(-1))}% CER at 8°`],
    ["LuWandSparkles", "Deskew restores accuracy", `${f1(cnnR.rotation_deskew.at(-1))}% at 8° with our preprocessing`],
    ["LuActivity", "Noise barely matters", "Heavy noise, 15% contrast or JPEG quality 8 change CER by under 0.2 points"],
    ["LuEye", "TrOCR is most tilt-robust", `${f1(robustness.models.trocr.rotation.at(-1))}% at 8°, but most blur-sensitive`],
  ];
  for (const [i, [ic, title, desc]] of insights.entries()) {
    const y = 1.55 + i * 1.27;
    await iconCircle(s, ic, 8.6, y, 0.62, { fill: C.background2 });
    text(s, title, { x: 9.4, y: y - 0.02, w: 3.35, h: 0.38, fontSize: 16, fontFace: THEME.headFontFace });
    text(s, desc, { x: 9.4, y: y + 0.36, w: 3.35, h: 0.75, fontSize: 13, color: C.accent6 });
  }
  s.addNotes("We re-read 80 test lines under controlled noise, blur, rotation, low contrast and JPEG compression. Noise and compression barely matter thanks to augmentation, but rotation breaks the CTC models and PaddleOCR. With our deskew step in front, the error returns almost to the clean value.");

  /* ---------------------------------------------------------------- 13 layout */
  pres.addSection({ title: "Application" });
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Application" });
  lens(s, -2.6, 4.3, 3.0);
  s.addText("Layout analysis: from page image to structure", { placeholder: "title" });
  const form = path.join(FIG, "screens", "s05_form_layout.jpg");
  s.addImage({ path: form, x: 0.6, y: 1.5, w: 7.9, h: 7.9 * 1350 / 2160, shadow: shadow(), objectName: "!!shot", altText: "Application form with layout overlay" });
  const kinds = [
    ["7A5AF8", "Titles and headings", "found from line height and stroke width"],
    ["2E90FA", "Text and key–value rows", "kept in reading order, column by column"],
    ["16B364", "Ruled tables", "rebuilt as Markdown tables"],
    ["F79009", "Checkboxes", "checked, empty or unsure"],
    ["EE46BC", "Signature fields", "from a printed label (not verification)"],
    ["F04438", "Low-confidence lines", "sent to the review queue"],
  ];
  kinds.forEach(([color, title, desc], i) => {
    const y = 1.55 + i * 0.83;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y: y + 0.08, w: 0.26, h: 0.26, rectRadius: 0.06, fill: { color }, line: { type: "none" } });
    text(s, title, { x: 9.4, y, w: 3.35, h: 0.38, fontSize: 16, fontFace: THEME.headFontFace });
    text(s, desc, { x: 9.4, y: y + 0.37, w: 3.35, h: 0.36, fontSize: 13, color: C.accent6 });
  });
  s.addNotes("After recognition, the layout is rebuilt from the detected lines and the page image. On this application form the system finds the title, the fields, the checkboxes and their state, the ruled table and the signature field, and writes them out as structured Markdown.");

  /* ---------------------------------------------------------------- 14 studio */
  s = pres.addSlide({ masterName: "Content light", sectionTitle: "Application" });
  lens(s, 11.5, -1.9, 3.2);
  s.addText("The DeepOCR studio: review instead of retyping", { placeholder: "title" });
  const studio = path.join(FIG, "screens", "s03_results_consensus.jpg");
  s.addImage({ path: studio, x: 4.75, y: 1.5, w: 7.95, h: 7.95 * 1350 / 2160, shadow: shadow(), objectName: "!!shot", altText: "Results workspace with a disputed word's alternatives" });
  const features = [
    ["LuActivity", "Live progress", "Every stage and every model, with cancel"],
    ["LuGitCompareArrows", "Disputed words", "One click swaps in another model's reading"],
    ["LuListChecks", "Review queue", "Unclear lines, tables, checkboxes, signatures"],
    ["LuDownload", "Export", "Markdown, plain text or JSON with coordinates"],
  ];
  for (const [i, [ic, title, desc]] of features.entries()) {
    const y = 1.55 + i * 1.22;
    await iconCircle(s, ic, 0.6, y, 0.62, { fill: C.background2 });
    text(s, title, { x: 1.4, y: y - 0.02, w: 3.1, h: 0.38, fontSize: 16, fontFace: THEME.headFontFace });
    text(s, desc, { x: 1.4, y: y + 0.36, w: 3.1, h: 0.7, fontSize: 13, color: C.accent6 });
  }
  text(s, "Also: Compare engines side by side, the Research results page, and a phone layout", { x: 0.6, y: 6.5, w: 4.0, h: 0.4, fontSize: 11, color: C.accent6 });
  s.addNotes("This is the results workspace. The page shows each line coloured by how much the models agreed. Clicking an underlined word shows what the other models read and their share of the vote, and one click replaces it. Every automatic fix can be undone, and the result exports as Markdown, text or JSON. This is a good point for the live demo.");

  /* ---------------------------------------------------------------- 15 conclusion */
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "Content dark", sectionTitle: "Conclusion" });
  lens(s, 8.25, 0.85, 5.8, { dark: true });
  [[2.5, 0], [1.75, 15], [2.1, 30]].forEach(([w, t], i) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 11.15 - 1.25, y: 2.45 + i * 0.42, w, h: 0.24, rectRadius: 0.12, objectName: `!!mark${i + 1}`,
    fill: { color: C.background1, transparency: t }, line: { type: "none" },
  }));
  text(s, "Thank you", { x: 8.25, y: 3.9, w: 5.8, h: 0.8, fontSize: 40, bold: true, color: C.background1, align: "center", fontFace: THEME.headFontFace });
  text(s, "Questions?", { x: 8.25, y: 4.7, w: 5.8, h: 0.5, fontSize: 20, color: SOFT_ON_DARK, align: "center" });
  s.addText("Conclusion and future work", { placeholder: "title" });
  text(s, "All eight proposal objectives met", { x: 0.6, y: 1.45, w: 7.2, h: 0.45, fontSize: 18, bold: true, color: SOFT_ON_DARK });
  text(s, [
    { text: "Two CTC models trained from scratch on a CPU; TrOCR fine-tuned (print CER 20.4% → 2.6%)", options: { bullet: true, breakLine: true } },
    { text: "Consensus vote: best accuracy (5.74% / 2.35% CER) and a guide for human review", options: { bullet: true, breakLine: true } },
    { text: "Complete pipeline and web studio, covered by 135 automated tests", options: { bullet: true } },
  ], { x: 0.6, y: 2.0, w: 7.2, h: 2.0, fontSize: 16, color: C.background1, paraSpaceAfter: 10, valign: "top" });
  text(s, "Future work", { x: 0.6, y: 4.35, w: 7.2, h: 0.4, fontSize: 16, bold: true, color: SOFT_ON_DARK });
  ["Hindi & multilingual", "Learned layout (LayoutLM)", "Faster, parallel voting", "Mobile camera OCR"].forEach((label, i) => {
    const x = 0.6 + (i % 2) * 3.65;
    const y = 4.85 + Math.floor(i / 2) * 0.68;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.45, h: 0.52, rectRadius: 0.26, fill: { color: C.background1, transparency: 88 }, line: { color: SOFT_ON_DARK, width: 0.75 } });
    text(s, label, { x, y, w: 3.45, h: 0.52, fontSize: 14, color: C.background1, align: "center", valign: "middle" });
  });
  s.addNotes("To conclude: all eight objectives of the proposal were met. The most effective improvement did not come from a bigger model but from letting the existing models vote, which both reduces errors and shows the reviewer where the remaining errors are likely to be. Future work includes Hindi and multilingual recognition, learned layout analysis, faster voting and a mobile app. Thank you.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(`Wrote ${OUT} (${pres.slides.length} slides)`);
}

build().catch((error) => {
  console.error(error);
  process.exit(1);
});
