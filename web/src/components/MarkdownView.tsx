import { Children, isValidElement, cloneElement, type ReactNode } from "react";
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
      if (part === "[Signature]") return <mark key={index} className="md-flag signature">✎ Signature</mark>;
      return part;
    });
  }
  if (Array.isArray(node)) return node.map((child, index) => <span key={index}>{decorate(child)}</span>);
  if (isValidElement<{ children?: ReactNode }>(node) && node.props.children) {
    return cloneElement(node, undefined, Children.map(node.props.children, decorate));
  }
  return node;
}

const withTokens = (Tag: "p" | "li" | "td" | "th" | "h1" | "h2" | "h3") =>
  function Decorated({ children }: { children?: ReactNode }) {
    return <Tag>{Children.map(children, decorate)}</Tag>;
  };

/** Renders OCR Markdown, turning review placeholders into visual chips. */
export function MarkdownView({ markdown }: { markdown: string }) {
  // Keep single line breaks from OCR output as visible lines.
  const source = markdown.replace(/([^\n|])\n(?=[^\n|#])/g, "$1  \n");
  return (
    <div className="markdown-view">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: withTokens("p"),
          li: withTokens("li"),
          td: withTokens("td"),
          th: withTokens("th"),
          h1: withTokens("h1"),
          h2: withTokens("h2"),
          h3: withTokens("h3"),
          a: ({ children }) => <span className="md-link">{children}</span>,
          img: () => null,
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
