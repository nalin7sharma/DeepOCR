/**
 * Build report/DeepOCR_Project_Report.docx from content.js.
 *
 *   cd report/src && npm install && node build_report.js
 *
 * Then run finalize_report.ps1 (Microsoft Word) to fill in the table of contents,
 * the lists of figures and tables, and to export a PDF copy.
 *
 * Inline markup in content strings: **bold**, *italic*, «monospace»,
 * {fig:id} / {tab:id} / {eq:id} cross-references (resolved to "Figure 4.2" etc.).
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");
const { meta, frontMatter, chapters, backMatter } = require("./content");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "report", "figures");
const OUT = path.join(ROOT, "report", "DeepOCR_Project_Report.docx");

const PAGE = { width: 11906, height: 16838 }; // A4
const MARGIN = { top: 1440, bottom: 1440, left: 2160, right: 1440, header: 720, footer: 720 };
const TEXT_W = PAGE.width - MARGIN.left - MARGIN.right; // 8306 twips
const TEXT_W_PX = (TEXT_W / 1440) * 96;
const FONT = "Times New Roman";
const MONO = "Consolas";
const BORDER = { style: D.BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const HEAD_FILL = "DCE3F0";
const HIGHLIGHT_FILL = "EEF2FF";

/* ------------------------------------------------------------ numbering pre-pass */
const refs = {};
function numberItems() {
  chapters.forEach((chapter) => {
    let fig = 0;
    let tab = 0;
    let eq = 0;
    chapter.items.forEach((item) => {
      if (item.type === "figure") item.label = `${chapter.num}.${++fig}`;
      if (item.type === "table") item.label = `${chapter.num}.${++tab}`;
      if (item.type === "eq") item.label = `${chapter.num}.${++eq}`;
      if (item.id) refs[`${item.type === "figure" ? "fig" : item.type === "table" ? "tab" : "eq"}:${item.id}`] = item.label;
    });
  });
}

function resolve(text) {
  return text.replace(/\{(fig|tab|eq):([\w-]+)\}/g, (match, kind, id) => {
    const label = refs[`${kind}:${id}`];
    if (!label) throw new Error(`Unknown reference ${match}`);
    return kind === "fig" ? `Figure ${label}` : kind === "tab" ? `Table ${label}` : `Equation (${label})`;
  });
}

/* ------------------------------------------------------------ inline runs */
function runs(text, base = {}) {
  const out = [];
  const tokens = resolve(text).split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|«[^»]+»)/);
  for (const token of tokens) {
    if (!token) continue;
    if (token.startsWith("**")) out.push(new D.TextRun({ ...base, text: token.slice(2, -2), bold: true }));
    else if (token.startsWith("«")) out.push(new D.TextRun({ ...base, text: token.slice(1, -1), font: MONO, size: Math.round((base.size ?? 24) * 0.85) }));
    else if (token.startsWith("*") && token.endsWith("*") && token.length > 2) out.push(new D.TextRun({ ...base, text: token.slice(1, -1), italics: true }));
    else out.push(new D.TextRun({ ...base, text: token }));
  }
  return out;
}

const body = (text, opts = {}) => new D.Paragraph({
  alignment: opts.align ?? D.AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 360 },
  keepNext: opts.keepNext,
  children: runs(text),
});

/* ------------------------------------------------------------ lists */
const numberingConfigs = [
  {
    reference: "bullets",
    levels: [
      { level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: D.LevelFormat.BULLET, text: "–", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 1260, hanging: 360 } } } },
    ],
  },
];
let listCount = 0;
function list(items, numbered) {
  let reference = "bullets";
  if (numbered) {
    reference = `numbers-${++listCount}`;
    numberingConfigs.push({
      reference,
      levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    });
  }
  return items.map((item) => {
    const nested = Array.isArray(item);
    return nested
      ? item.map((sub) => new D.Paragraph({ numbering: { reference: "bullets", level: 1 }, spacing: { after: 60, line: 312 }, alignment: D.AlignmentType.LEFT, children: runs(sub) }))
      : [new D.Paragraph({ numbering: { reference, level: 0 }, spacing: { after: 80, line: 324 }, alignment: D.AlignmentType.JUSTIFIED, children: runs(item) })];
  }).flat();
}

/* ------------------------------------------------------------ figures */
function imageSize(file) {
  const buf = fs.readFileSync(file);
  if (buf.readUInt32BE(0) === 0x89504e47) return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20), type: "png", buf };
  let i = 2;
  while (i < buf.length) {
    const marker = buf[i + 1];
    const length = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
      return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7), type: "jpg", buf };
    }
    i += 2 + length;
  }
  throw new Error(`Cannot read image size of ${file}`);
}

function figure(item) {
  const img = imageSize(path.join(FIG, item.file));
  let width = TEXT_W_PX * (item.width ?? 1);
  let height = (width * img.height) / img.width;
  const maxHeight = item.maxHeight ?? 610;
  if (height > maxHeight) {
    width *= maxHeight / height;
    height = maxHeight;
  }
  return [
    new D.Paragraph({
      alignment: D.AlignmentType.CENTER,
      spacing: { before: 120, after: 80 },
      keepNext: true,
      children: [new D.ImageRun({ type: img.type, data: img.buf, transformation: { width: Math.round(width), height: Math.round(height) } })],
    }),
    new D.Paragraph({
      style: "FigureCaption",
      children: [new D.TextRun({ text: `Figure ${item.label}: `, bold: true }), ...runs(item.caption)],
    }),
  ];
}

/* ------------------------------------------------------------ tables */
function cell(text, width, { header, align, fill, bold, size }) {
  return new D.TableCell({
    width: { size: width, type: D.WidthType.DXA },
    shading: fill ? { fill, type: D.ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: D.VerticalAlign.CENTER,
    children: String(text).split("\n").map((line) => new D.Paragraph({
      alignment: align === "r" ? D.AlignmentType.RIGHT : align === "c" ? D.AlignmentType.CENTER : D.AlignmentType.LEFT,
      spacing: { after: 0, line: 252 },
      children: runs(line, { size: size ?? 19, bold: header || bold }),
    })),
  });
}

function table(item) {
  const total = item.widths.reduce((a, b) => a + b, 0);
  const widths = item.widths.map((w) => Math.floor((w / total) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const align = item.align ?? item.headers.map((_, i) => (i === 0 ? "l" : "l"));
  const highlight = new Set(item.highlight ?? []);
  const rows = [
    new D.TableRow({
      tableHeader: true,
      cantSplit: true,
      children: item.headers.map((h, i) => cell(h, widths[i], { header: true, align: align[i], fill: HEAD_FILL, size: item.size })),
    }),
    ...item.rows.map((row, r) => new D.TableRow({
      cantSplit: true,
      children: row.map((value, i) => cell(value, widths[i], {
        align: align[i],
        fill: highlight.has(r) ? HIGHLIGHT_FILL : undefined,
        bold: highlight.has(r) && item.boldHighlight !== false,
        size: item.size,
      })),
    })),
  ];
  const caption = item.noCaption ? [] : [new D.Paragraph({
    style: "TableCaption",
    children: [new D.TextRun({ text: `Table ${item.label}: `, bold: true }), ...runs(item.caption)],
  })];
  return [
    ...caption,
    new D.Table({
      width: { size: TEXT_W, type: D.WidthType.DXA },
      columnWidths: widths,
      borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER, insideHorizontal: BORDER, insideVertical: BORDER },
      rows,
    }),
    new D.Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

/* ------------------------------------------------------------ equations and code */
function equation(item) {
  return new D.Paragraph({
    tabStops: [{ type: D.TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: D.TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 120, after: 160 },
    children: [
      new D.TextRun({ children: [new D.Tab()] }),
      ...runs(item.text, { italics: true }),
      new D.TextRun({ children: [new D.Tab(), `(${item.label})`] }),
    ],
  });
}

function code(item) {
  // About 80 monospace characters fit across the text block; longer lines would wrap.
  item.lines.filter((line) => [...line].length > 80).forEach((line) => console.warn(`Code line over 80 chars: ${line}`));
  const lines = item.lines.map((line) => new D.Paragraph({
    spacing: { after: 0, line: 240 },
    children: [new D.TextRun({ text: line.length ? line : " ", font: MONO, size: 17 })],
  }));
  const out = [];
  if (item.title) {
    out.push(new D.Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: runs(item.title, { bold: true, size: 21 }) }));
  }
  out.push(new D.Table({
    width: { size: TEXT_W, type: D.WidthType.DXA },
    columnWidths: [TEXT_W],
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER, insideHorizontal: BORDER, insideVertical: BORDER },
    rows: [new D.TableRow({
      children: [new D.TableCell({
        width: { size: TEXT_W, type: D.WidthType.DXA },
        shading: { fill: "F5F6F8", type: D.ShadingType.CLEAR, color: "auto" },
        margins: { top: 100, bottom: 100, left: 160, right: 160 },
        children: lines,
      })],
    })],
  }));
  out.push(new D.Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

/* ------------------------------------------------------------ items */
function render(item) {
  switch (item.type) {
    case "p": return [body(item.text, item)];
    case "h2": return [new D.Paragraph({ heading: D.HeadingLevel.HEADING_2, keepNext: true, children: runs(item.text) })];
    case "h3": return [new D.Paragraph({ heading: D.HeadingLevel.HEADING_3, keepNext: true, children: runs(item.text) })];
    case "bullets": return list(item.items, false);
    case "numbers": return list(item.items, true);
    case "figure": return figure(item);
    case "table": return table(item);
    case "eq": return [equation(item)];
    case "code": return code(item);
    case "pagebreak": return [new D.Paragraph({ children: [new D.PageBreak()] })];
    default: throw new Error(`Unknown item type ${item.type}`);
  }
}

function chapterHeading(text, first = true) {
  return new D.Paragraph({ heading: D.HeadingLevel.HEADING_1, pageBreakBefore: first, children: runs(text) });
}

/* ------------------------------------------------------------ document */
numberItems();

/** Names and roll numbers on the title page: a centred two-column table without borders. */
function teamTable() {
  const none = { style: D.BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const widths = [3600, 2800];
  const left = Math.round((TEXT_W - widths[0] - widths[1]) / 2);
  return new D.Table({
    width: { size: widths[0] + widths[1], type: D.WidthType.DXA },
    columnWidths: widths,
    indent: { size: left, type: D.WidthType.DXA },
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: meta.team.map(([name, roll]) => new D.TableRow({
      children: [[name, false], [roll, true]].map(([value, isRoll], i) => new D.TableCell({
        width: { size: widths[i], type: D.WidthType.DXA },
        borders: { top: none, bottom: none, left: none, right: none },
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new D.Paragraph({
          alignment: isRoll ? D.AlignmentType.RIGHT : D.AlignmentType.LEFT,
          spacing: { after: 0, line: 276 },
          children: [new D.TextRun({ text: value, font: FONT, size: 24, bold: !isRoll, color: isRoll ? "475467" : "101828" })],
        })],
      })),
    })),
  });
}

const titlePage = [
  new D.Paragraph({ spacing: { before: 1400, after: 240 }, alignment: D.AlignmentType.CENTER, children: [new D.TextRun({ text: "PROJECT REPORT", font: FONT, size: 26, bold: true, characterSpacing: 60, color: "4F46E5" })] }),
  new D.Paragraph({ spacing: { after: 360 }, alignment: D.AlignmentType.CENTER, children: [new D.TextRun({ text: meta.title, font: FONT, size: 40, bold: true })] }),
  new D.Paragraph({
    alignment: D.AlignmentType.CENTER,
    spacing: { after: 900 },
    border: { bottom: { style: D.BorderStyle.SINGLE, size: 8, color: "4F46E5", space: 12 } },
    children: [new D.TextRun({ text: meta.subtitle, font: FONT, size: 26, italics: true, color: "475467" })],
  }),
  new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 160 }, children: [new D.TextRun({ text: "Submitted by", font: FONT, size: 24, bold: true })] }),
  teamTable(),
  new D.Paragraph({ spacing: { after: 900 }, children: [] }),
  ...meta.titleLines.map((line) => new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 120 }, children: [new D.TextRun({ text: line, font: FONT, size: 24 })] })),
];

const front = [];
frontMatter.forEach((section, index) => {
  if (section.placeholder) {
    front.push(new D.Paragraph({ style: "FrontTitle", pageBreakBefore: index > 0, children: [new D.TextRun(section.title)] }));
    front.push(new D.Paragraph({ children: [new D.TextRun(section.placeholder)] }));
    return;
  }
  front.push(chapterHeading(section.title, index > 0));
  section.items.forEach((item) => front.push(...render(item)));
});

const main = [];
chapters.forEach((chapter) => {
  main.push(chapterHeading(`${chapter.num}. ${chapter.title}`));
  chapter.items.forEach((item) => main.push(...render(item)));
});
backMatter.forEach((section) => {
  main.push(chapterHeading(section.title));
  section.items.forEach((item) => {
    if (item.type === "refs") {
      item.items.forEach((ref, i) => main.push(new D.Paragraph({
        spacing: { after: 100, line: 300 },
        indent: { left: 567, hanging: 567 },
        alignment: D.AlignmentType.LEFT,
        children: [new D.TextRun({ text: `[${i + 1}]\t` }), ...runs(ref)],
        tabStops: [{ type: D.TabStopType.LEFT, position: 567 }],
      })));
    } else {
      main.push(...render(item));
    }
  });
});

const pageNumberFooter = new D.Footer({
  children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER, children: [new D.TextRun({ children: [D.PageNumber.CURRENT], size: 20 })] })],
});
const runningHeader = new D.Header({
  children: [new D.Paragraph({
    alignment: D.AlignmentType.RIGHT,
    border: { bottom: { style: D.BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
    children: [new D.TextRun({ text: meta.runningHead, italics: true, size: 18, color: "595959" })],
  })],
});

const doc = new D.Document({
  creator: meta.author || "DeepOCR",
  title: meta.title,
  description: meta.subtitle,
  styles: {
    default: {
      document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: 360 } } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 34, bold: true, color: "1F2A44" },
        paragraph: { spacing: { before: 0, after: 360, line: 300 }, outlineLevel: 0, border: { bottom: { style: D.BorderStyle.SINGLE, size: 8, color: "4F46E5", space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "1F2A44" },
        paragraph: { spacing: { before: 300, after: 120, line: 300 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "1F2A44" },
        paragraph: { spacing: { before: 200, after: 80, line: 300 }, outlineLevel: 2 } },
      { id: "FrontTitle", name: "Front Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 34, bold: true, color: "1F2A44" },
        paragraph: { spacing: { after: 360, line: 300 }, border: { bottom: { style: D.BorderStyle.SINGLE, size: 8, color: "4F46E5", space: 6 } } } },
      { id: "FigureCaption", name: "Figure Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 21 },
        paragraph: { alignment: D.AlignmentType.CENTER, spacing: { before: 0, after: 240, line: 276 } } },
      { id: "TableCaption", name: "Table Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 21 },
        paragraph: { alignment: D.AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100, line: 276 } } },
    ],
  },
  numbering: { config: numberingConfigs },
  sections: [
    {
      properties: { page: { size: PAGE, margin: MARGIN } },
      children: titlePage,
    },
    {
      properties: { page: { size: PAGE, margin: MARGIN, pageNumbers: { start: 1, formatType: D.NumberFormat.LOWER_ROMAN } } },
      footers: { default: pageNumberFooter },
      children: front,
    },
    {
      properties: { page: { size: PAGE, margin: MARGIN, pageNumbers: { start: 1, formatType: D.NumberFormat.DECIMAL } } },
      headers: { default: runningHeader },
      footers: { default: pageNumberFooter },
      children: main,
    },
  ],
});

D.Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  const figures = chapters.flatMap((c) => c.items).filter((i) => i.type === "figure").length;
  const tables = chapters.flatMap((c) => c.items).filter((i) => i.type === "table").length;
  console.log(`Wrote ${OUT} (${chapters.length} chapters, ${figures} figures, ${tables} tables)`);
});
