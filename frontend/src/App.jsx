import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  Building2,
  ImagePlus,
  LineChart,
  Loader2,
  Menu,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import { MarketChart } from "./components/MarketChart";
import { searchCompanies, sendChat } from "./lib/api";
import "./styles.css";

const starters = [
  "Analyze RELIANCE using fundamentals and 6 month technicals",
  "What is the RSI and trend for TCS?",
  "Compare the risks in this company before earnings",
  "Give me a swing trading view on HDFCBANK",
];

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderInlineMarkdown(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function isMarketQuery(text, company, symbol) {
  const haystack = `${text} ${company} ${symbol}`.toLowerCase();
  return /\b(stock|share|nse|bse|market|finance|financial|technical|fundamental|rsi|ema|macd|atr|price|earnings|revenue|profit|loss|balance sheet|cash flow|trading|invest|valuation|risk)\b/.test(haystack);
}

function Message({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "assistant" ? <Bot size={18} /> : "You"}</div>
      <div className="bubble">
        {message.attachments?.length > 0 && (
          <div className="message-attachments">
            {message.attachments.map((file) => (
              file.mimeType?.startsWith("image/") ? (
                <img key={file.name} src={file.dataUrl} alt={file.name} />
              ) : (
                <span key={file.name}>{file.name}</span>
              )
            ))}
          </div>
        )}
        {message.content.split("\n").map((line, index) => (
          <p key={index}>{line ? renderInlineMarkdown(line) : "\u00A0"}</p>
        ))}
      </div>
    </article>
  );
}

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask me about an NSE stock. I can combine your fundamental documents with Yahoo Finance technical data.",
    },
  ]);
  const [input, setInput] = useState("");
  const [company, setCompany] = useState("");
  const [symbol, setSymbol] = useState("");
  const [period, setPeriod] = useState("6mo");
  const [interval, setInterval] = useState("1d");
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [attachments, setAttachments] = useState([]);
  const [latestTechnicals, setLatestTechnicals] = useState(null);
  const fileInput = useRef(null);

  const history = useMemo(
    () =>
      messages
        .filter((message) => message.role !== "system")
        .slice(-6)
        .map((message) => ({
          ...message,
          content:
            message.content.length > 1200
              ? `${message.content.slice(0, 1200)}...`
              : message.content,
        })),
    [messages],
  );

  async function handleCompanyChange(value) {
    setCompany(value);
    if (value.length < 2) {
      setSuggestions([]);
      return;
    }
    setSuggestions(await searchCompanies(value));
  }

  async function handleFiles(event) {
    const files = Array.from(event.target.files || []);
    const mapped = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        mimeType: file.type || "application/octet-stream",
        dataUrl: await fileToDataUrl(file),
      })),
    );
    setAttachments((items) => [...items, ...mapped]);
    event.target.value = "";
  }

  async function submit(text = input) {
    const message = text.trim();
    const selectedAttachments = attachments;
    if ((!message && selectedAttachments.length === 0) || busy) return;
    const outgoingText = message || "Please analyze the attached image.";
    setInput("");
    setAttachments([]);
    setBusy(true);
    setMessages((items) => [
      ...items,
      {
        role: "user",
        content: outgoingText,
        attachments: selectedAttachments,
      },
    ]);
    try {
      const response = await sendChat({
        message: outgoingText,
        history,
        attachments: selectedAttachments,
        company: company || null,
        symbol: symbol || null,
        period,
        interval,
      });
      setLatestTechnicals(response.technicals?.candles ? response.technicals : null);
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: response.answer,
        },
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        { role: "assistant", content: error.message },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span><Sparkles size={18} /></span>
          <strong>India Market AI</strong>
        </div>
        <button className="new-chat" onClick={() => setMessages([])}>
          <Menu size={16} /> New chat
        </button>
        <div className="field">
          <label><Building2 size={15} /> Company</label>
          <input
            value={company}
            onChange={(event) => handleCompanyChange(event.target.value)}
            placeholder="Reliance Industries"
          />
          {suggestions.length > 0 && (
            <div className="suggestions">
              {suggestions.map((item) => (
                <button
                  key={item.fundamentals_url}
                  onClick={() => {
                    setCompany(item.company);
                    setSuggestions([]);
                  }}
                >
                  {item.company}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="field">
          <label><LineChart size={15} /> Yahoo symbol</label>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="RELIANCE" />
        </div>
        <div className="row">
          <select value={period} onChange={(e) => setPeriod(e.target.value)}>
            <option value="1mo">1M</option>
            <option value="3mo">3M</option>
            <option value="6mo">6M</option>
            <option value="1y">1Y</option>
            <option value="5y">5Y</option>
          </select>
          <select value={interval} onChange={(e) => setInterval(e.target.value)}>
            <option value="1d">1D</option>
            <option value="1wk">1W</option>
            <option value="1mo">1M</option>
          </select>
        </div>
        {latestTechnicals && <MarketChart technicals={latestTechnicals} />}
      </aside>

      <section className="chat">
        <div className="thread">
          {messages.length === 0 ? (
            <div className="empty">
              <span><Sparkles size={24} /></span>
              <h1>How can I help with the Indian market?</h1>
              <div className="starter-grid">
                {starters.map((starter) => (
                  <button key={starter} onClick={() => submit(starter)}>
                    {starter}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => <Message key={index} message={message} />)
          )}
          {busy && (
            <article className="message assistant">
              <div className="avatar"><Bot size={18} /></div>
              <div className="bubble muted">
                <Loader2 className="spin" size={18} />
                {isMarketQuery(input || messages.at(-1)?.content || "", company, symbol)
                  ? "Analyzing market data"
                  : "Thinking"}
              </div>
            </article>
          )}
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          {attachments.length > 0 && (
            <div className="attachments">
              {attachments.map((file) => (
                <span key={file.name}>
                  {file.name}
                  <button type="button" onClick={() => setAttachments((items) => items.filter((x) => x.name !== file.name))}>
                    <X size={13} />
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="composer-row">
            <button type="button" className="icon-btn" onClick={() => fileInput.current?.click()} title="Attach image">
              <ImagePlus size={20} />
            </button>
            <input ref={fileInput} type="file" accept="image/*,.pdf" multiple hidden onChange={handleFiles} />
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submit();
                }
              }}
              placeholder="Ask about fundamentals, RSI, trend, risks, or upload a chart image"
              rows={1}
            />
            <button className="send" disabled={busy || (!input.trim() && attachments.length === 0)} title="Send">
              {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
