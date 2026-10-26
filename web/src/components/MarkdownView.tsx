import { Children, cloneElement, isValidElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const TOKEN = /(\[Illegible\]|\[Signature\]|\[x\]|\[ \]|\[\?\])/g;

function decorate(node: ReactNode): ReactNode {
  if (typeof node === "string") {
    const parts = node.split(TOKEN);
    if (parts.length === 1) return node;
    return parts.map((part, index) => {
      if (part === "[x]") return <span key={index} className="md-box checked" title="Checked box">✓</span>;
      if (part === "[ ]") return <span key={index} className="md-box" title="Unchecked box" />;
      if (part === "[?]") return <span key={index} className="md-box uncertain" title="Unclear box">?</span>;
      if (part === "[Illegible]") return <mark key={index} className="md-flag illegible">Illegible</mark>;
      if (part === "[Signature]") return <mark key={index} className="md-flag signature">✎ Signature field</mark>;
      return part;
    });
  }
  if (isValidElement<{ children?: ReactNode }>(node) && node.props.children) {
    return cloneElement(node, undefined, Children.map(node.props.children, decorate));
  }
  return node;
}

const withTokens = (Tag: "p" | "li" | "td" | "th" | "h1" | "h3") =>
  function Decorated({ children }: { children?: ReactNode }) {
    return <Tag>{Children.map(children, decorate)}</Tag>;
  };

function Heading2({ children }: { children?: ReactNode }) {
  const text = Children.toArray(children).join("");
  if (/^Page \d+$/.test(text)) return <h2 className="page-h">{text}</h2>;
  return <h2>{Children.map(children, decorate)}</h2>;
}

/** Renders OCR Markdown, turning review placeholders into visual chips. */
export function MarkdownView({ markdown }: { markdown: string }) {
  // Keep OCR line breaks visible (Markdown would otherwise join them).
  const source = markdown.replace(/([^\n|])\n(?=[^\n|#])/g, "$1  \n");
  return (
    <div className="doc-text">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: withTokens("p"),
          li: withTokens("li"),
          td: withTokens("td"),
          th: withTokens("th"),
          h1: withTokens("h1"),
          h2: Heading2,
          h3: withTokens("h3"),
          a: ({ children }) => <span>{children}</span>,
          img: () => null,
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
