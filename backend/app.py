from __future__ import annotations

import base64
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import httpx
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QYs69eiWyYO6IireGHEl1HZn6um69gXrYOE_PjlBNLM/edit?usp=sharing"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
FUNDAMENTALS_CONTEXT_CHARS = int(os.getenv("FUNDAMENTALS_CONTEXT_CHARS", "6000"))
HISTORY_MESSAGE_CHARS = int(os.getenv("HISTORY_MESSAGE_CHARS", "600"))
FUNDAMENTALS_SHEET_URL = os.getenv("FUNDAMENTALS_SHEET_URL", DEFAULT_SHEET_URL)
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent


class Attachment(BaseModel):
    name: str
    mime_type: str = Field(alias="mimeType")
    data_url: str = Field(alias="dataUrl")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    attachments: list[Attachment] = []
    symbol: str | None = None
    company: str | None = None
    period: str = "6mo"
    interval: str = "1d"


class CompanyMatch(BaseModel):
    company: str
    fundamentals_url: str


class ChatResponse(BaseModel):
    answer: str
    company_match: CompanyMatch | None = None
    technicals: dict[str, Any] | None = None
    sources: list[str] = []


class AgentState(TypedDict, total=False):
    request: ChatRequest
    company_match: dict[str, str] | None
    fundamentals_text: str
    fundamentals_data: dict[str, Any] | None
    technicals: dict[str, Any] | None
    use_market_tools: bool
    answer: str
    sources: list[str]


app = FastAPI(title="India Market AI", version="0.1.0")
frontend_origins = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def spreadsheet_id_from_url(sheet_url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("Invalid Google Sheet URL")
    return match.group(1)


@lru_cache(maxsize=1)
def load_fundamentals_index() -> pd.DataFrame:
    spreadsheet = gspread_client().open_by_key(spreadsheet_id_from_url(FUNDAMENTALS_SHEET_URL))
    values = spreadsheet.sheet1.get_all_values()
    df = pd.DataFrame(trim_grid(values))
    df = df.iloc[:, :2].dropna()
    df.columns = ["company", "fundamentals_url"]
    df["company"] = df["company"].astype(str).str.strip()
    df["fundamentals_url"] = df["fundamentals_url"].astype(str).str.strip()
    df = df[df["company"].str.len() > 0]
    return df[df["company"].str.lower() != "stock name"]


def find_company(query: str | None, message: str) -> dict[str, str] | None:
    search_text = (query or message).lower()
    df = load_fundamentals_index()

    exact = df[df["company"].str.lower() == search_text.strip()]
    if not exact.empty:
        row = exact.iloc[0]
        return {"company": row.company, "fundamentals_url": row.fundamentals_url}

    contains = df[df["company"].str.lower().apply(lambda name: name in search_text)]
    if contains.empty and query:
        contains = df[df["company"].str.lower().str.contains(re.escape(query.lower()), na=False)]
    if contains.empty:
        return None
    row = contains.iloc[0]
    return {"company": row.company, "fundamentals_url": row.fundamentals_url}


def google_doc_export_url(url: str) -> str | None:
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return None
    return f"https://docs.google.com/document/d/{match.group(1)}/export?format=txt"


def fetch_document_text(url: str) -> str:
    export_url = google_doc_export_url(url) or url
    import httpx

    response = httpx.get(export_url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.text.strip()[:8000]


def gspread_client() -> Any:
    try:
        import gspread
    except ImportError as exc:
        raise RuntimeError("gspread is not installed. Run pip install -r backend/requirements.txt.") from exc

    service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if service_account_file:
        credential_path = resolve_credential_path(service_account_file)
        return gspread.service_account(filename=str(credential_path))
    raise RuntimeError(
        "Google Sheets credentials are not configured. Set GOOGLE_APPLICATION_CREDENTIALS."
    )


def resolve_credential_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path.cwd() / path,
        APP_DIR / path,
        PROJECT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Google credentials file was not found. Checked: {checked}")


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def trim_grid(values: list[list[Any]]) -> list[list[str]]:
    rows = [[clean_cell(cell) for cell in row] for row in values]
    while rows and not any(rows[-1]):
        rows.pop()
    width = max((len(row) for row in rows), default=0)
    padded = [row + [""] * (width - len(row)) for row in rows]
    while padded and width and not any(row[width - 1] for row in padded):
        width -= 1
    return [row[:width] for row in padded]


def parse_number(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned in {"-", "--", "NA", "N/A"}:
        return None
    multiplier = 1
    if cleaned.startswith("(") and cleaned.endswith(")"):
        multiplier = -1
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace(",", "").replace("%", "").replace("₹", "").replace("$", "")
    cleaned = re.sub(r"\s*(cr|crore|mn|m|bn|b)$", "", cleaned, flags=re.I)
    try:
        return round(float(cleaned) * multiplier, 4)
    except ValueError:
        return None


def is_period_label(value: str) -> bool:
    label = value.strip()
    return bool(
        re.search(r"\b(?:19|20)\d{2}\b", label)
        or re.search(r"\bQ[1-4]\b", label, flags=re.I)
        or re.search(r"\bFY\s*\d{2,4}\b", label, flags=re.I)
        or re.search(r"\bTTM\b", label, flags=re.I)
    )


def likely_header_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell]
    if len(non_empty) < 2:
        return False
    return sum(1 for cell in row[1:] if is_period_label(cell)) >= 1


def row_is_mostly_empty(row: list[str]) -> bool:
    return sum(1 for cell in row if cell) <= 1


def normalize_worksheet(title: str, values: list[list[Any]]) -> dict[str, Any]:
    grid = trim_grid(values)
    tables: list[dict[str, Any]] = []
    validation: list[str] = []
    row_index = 0

    while row_index < len(grid):
        row = grid[row_index]
        if not likely_header_row(row):
            row_index += 1
            continue

        header_row_number = row_index + 1
        headers = [cell or f"Column {idx + 1}" for idx, cell in enumerate(row)]
        table_rows: list[dict[str, Any]] = []
        metrics: dict[str, dict[str, float | str | None]] = {}
        row_index += 1

        while row_index < len(grid):
            current = grid[row_index]
            if likely_header_row(current):
                break
            if row_is_mostly_empty(current):
                row_index += 1
                if table_rows:
                    break
                continue

            metric_name = current[0] or f"Row {row_index + 1}"
            record: dict[str, Any] = {
                "row_number": row_index + 1,
                "metric": metric_name,
                "values": {},
            }
            for col_index, header in enumerate(headers[1:], start=1):
                raw = current[col_index] if col_index < len(current) else ""
                parsed = parse_number(raw)
                record["values"][header] = parsed if parsed is not None else raw
            table_rows.append(record)
            metrics[metric_name] = record["values"]
            row_index += 1

        if table_rows:
            tables.append(
                {
                    "header_row_number": header_row_number,
                    "headers": headers,
                    "rows": table_rows,
                    "metrics": metrics,
                }
            )
        else:
            validation.append(f"Tab '{title}' has a header-like row with no data rows.")

    if not tables and grid:
        validation.append(f"Tab '{title}' did not contain a recognizable metric-by-period table.")

    return {
        "title": title,
        "row_count": len(grid),
        "column_count": len(grid[0]) if grid else 0,
        "tables": tables,
        "validation": validation,
    }


def load_structured_spreadsheet(url: str) -> dict[str, Any]:
    spreadsheet_id = spreadsheet_id_from_url(url)
    spreadsheet = gspread_client().open_by_key(spreadsheet_id)
    tabs = []
    validation = []

    for worksheet in spreadsheet.worksheets():
        values = worksheet.get_all_values()
        normalized = normalize_worksheet(worksheet.title, values)
        tabs.append(normalized)
        validation.extend(normalized["validation"])

    expected_tabs = ["Profit & Loss","Quarters", "Balance Sheet", "Cash Flow", "Data Sheet"]
    available_titles = {tab["title"].lower() for tab in tabs}
    for tab_name in expected_tabs:
        if tab_name.lower() not in available_titles:
            validation.append(f"Expected tab missing or differently named: {tab_name}")

    return {
        "spreadsheet_id": spreadsheet_id,
        "source_url": url,
        "tabs": tabs,
        "validation": validation,
    }


def fetch_fundamentals_data(url: str) -> dict[str, Any]:
    if google_doc_export_url(url):
        return {
            "source_url": url,
            "document_text": fetch_document_text(url),
            "validation": ["Source is a Google Doc, not a structured spreadsheet."],
        }
    if re.search(r"/spreadsheets/d/", url):
        return load_structured_spreadsheet(url)
    return {
        "source_url": url,
        "document_text": fetch_document_text(url),
        "validation": ["Source is not a Google Sheet URL; loaded as text only."],
    }


def to_yahoo_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if "." in cleaned or cleaned.startswith("^"):
        return cleaned
    if cleaned in {"NIFTY", "NIFTY50"}:
        return "^NSEI"
    if cleaned in {"BANKNIFTY", "NIFTYBANK"}:
        return "^NSEBANK"
    return f"{cleaned}.NS"


def normalize_yahoo_period(period: str) -> str:
    normalized = period.strip().lower()
    aliases = {
        "1d": "1d",
        "5d": "5d",
        "1mo": "1mo",
        "1m": "1mo",
        "3mo": "3mo",
        "3m": "3mo",
        "6mo": "6mo",
        "6m": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
        "ytd": "ytd",
        "max": "max",
    }
    return aliases.get(normalized, "6mo")


def normalize_yahoo_interval(interval: str) -> str:
    normalized = interval.strip().lower()
    aliases = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "90m": "90m",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1w": "1wk",
        "1mo": "1mo",
        "1mth": "1mo",
        "3mo": "3mo",
    }
    return aliases.get(normalized, "1d")


def compute_technicals(symbol: str, period: str, interval: str) -> dict[str, Any]:
    yf_symbol = to_yahoo_symbol(symbol)
    yahoo_period = normalize_yahoo_period(period)
    yahoo_interval = normalize_yahoo_interval(interval)
    errors: list[str] = []

    try:
        history = yf.Ticker(yf_symbol).history(
            period=yahoo_period,
            interval=yahoo_interval,
            auto_adjust=False,
        )
    except Exception as exc:
        errors.append(f"{yf_symbol} history failed: {exc}")
        history = pd.DataFrame()

    if history.empty:
        try:
            history = yf.download(
                yf_symbol,
                period=yahoo_period,
                interval=yahoo_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            errors.append(f"{yf_symbol} download failed: {exc}")
            history = pd.DataFrame()

    if history.empty and yf_symbol.endswith(".NS"):
        yf_symbol = yf_symbol[:-3] + ".BO"
        try:
            history = yf.download(
                yf_symbol,
                period=yahoo_period,
                interval=yahoo_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            errors.append(f"{yf_symbol} download failed: {exc}")
            history = pd.DataFrame()

    if history.empty:
        detail = "; ".join(errors) if errors else "Yahoo returned an empty price history."
        raise ValueError(
            f"No Yahoo Finance data found for {to_yahoo_symbol(symbol)} or BSE fallback. {detail}"
        )

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = history.dropna(subset=["Open", "High", "Low", "Close"])
    if history.empty:
        raise ValueError(
            f"Yahoo Finance returned rows for {yf_symbol}, but OHLC columns were empty after cleaning."
        )

    close = history["Close"]
    high = history["High"]
    low = history["Low"]
    volume = history["Volume"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()

    last = history.iloc[-1]
    previous_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
    change_pct = ((last["Close"] - previous_close) / previous_close) * 100
    recent_rows = history.tail(80).reset_index()
    candles = [
        {
            "time": str(row.iloc[0]),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
        }
        for _, row in recent_rows.iterrows()
    ]

    return {
        "symbol": symbol,
        "yahoo_symbol": yf_symbol,
        "period": yahoo_period,
        "interval": yahoo_interval,
        "last_close": round(float(last["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume": int(last["Volume"]) if pd.notna(last["Volume"]) else 0,
        "rsi_14": round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else None,
        "ema_20": round(float(ema_20.iloc[-1]), 2),
        "ema_50": round(float(ema_50.iloc[-1]), 2),
        "macd": round(float(macd.iloc[-1]), 2),
        "macd_signal": round(float(signal.iloc[-1]), 2),
        "atr_14": round(float(atr.iloc[-1]), 2) if pd.notna(atr.iloc[-1]) else None,
        "candles": candles,
    }


def maybe_symbol(request: ChatRequest) -> str | None:
    if request.symbol:
        return request.symbol
    match = re.search(r"\b([A-Z]{2,12})(?:\.NS|\.BO)?\b", request.message.upper())
    if match:
        return match.group(1)
    if request.company:
        company = re.sub(r"\b(LTD|LIMITED|INDUSTRIES|INDIA|CORP|CORPORATION|COMPANY)\b", "", request.company.upper())
        first_word = re.findall(r"[A-Z]{3,12}", company)
        if first_word:
            return first_word[0]
    return None


def has_image_attachment(request: ChatRequest) -> bool:
    return any(item.mime_type.startswith("image/") for item in request.attachments)


def is_market_query(request: ChatRequest) -> bool:
    text = f"{request.message} {request.company or ''} {request.symbol or ''}".lower()
    market_terms = {
        "stock",
        "share",
        "nse",
        "bse",
        "market",
        "finance",
        "financial",
        "technical",
        "fundamental",
        "rsi",
        "ema",
        "macd",
        "atr",
        "price",
        "earnings",
        "revenue",
        "profit",
        "loss",
        "balance sheet",
        "cash flow",
        "trading",
        "invest",
        "valuation",
        "risk",
        "company",
    }
    if any(term in text for term in market_terms):
        return True
    if request.symbol:
        return True
    if request.company and re.search(r"\b(this|their|its|stock|company)\b", request.message.lower()):
        return True
    return False


def tool_node(state: AgentState) -> AgentState:
    request = state["request"]
    sources: list[str] = []
    company_match = None
    fundamentals_text = ""
    fundamentals_data = None
    use_market_tools = is_market_query(request)

    if use_market_tools:
        try:
            company_match = find_company(request.company, request.message)
        except Exception as exc:
            fundamentals_text = f"Company index could not be loaded: {exc}"
            fundamentals_data = {
                "error": str(exc),
                "validation": ["Company index could not be loaded."],
            }

    if use_market_tools and company_match:
        sources.append(company_match["fundamentals_url"])
        try:
            fundamentals_data = fetch_fundamentals_data(company_match["fundamentals_url"])
        except Exception as exc:
            fundamentals_text = f"Fundamental source could not be fetched: {exc}"
            fundamentals_data = {
                "source_url": company_match["fundamentals_url"],
                "error": str(exc),
                "validation": ["Fundamental source could not be fetched."],
            }

    technicals = None
    symbol = maybe_symbol(request) if use_market_tools else None
    if symbol:
        try:
            technicals = compute_technicals(symbol, request.period, request.interval)
            sources.append(f"https://finance.yahoo.com/quote/{technicals['yahoo_symbol']}")
        except Exception as exc:
            technicals = {"symbol": symbol, "error": str(exc)}

    return {
        **state,
        "company_match": company_match,
        "fundamentals_text": fundamentals_text,
        "fundamentals_data": fundamentals_data,
        "technicals": technicals,
        "use_market_tools": use_market_tools,
        "sources": sources,
    }


def attachment_summary(attachments: list[Attachment]) -> str:
    if not attachments:
        return "No image or file attachments."
    lines = []
    for item in attachments:
        size = len(base64.b64decode(item.data_url.split(",", 1)[-1])) if "," in item.data_url else 0
        lines.append(f"- {item.name} ({item.mime_type}, approx {size} bytes)")
    return "\n".join(lines)


def compact_technicals(technicals: dict[str, Any] | None) -> dict[str, Any] | str:
    if not technicals:
        return "No technical data loaded."
    if technicals.get("error"):
        return technicals
    return {
        key: value
        for key, value in technicals.items()
        if key != "candles"
    }


def compact_fundamentals(data: dict[str, Any] | None) -> str:
    if not data:
        return "No fundamental document matched or loaded."
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) <= FUNDAMENTALS_CONTEXT_CHARS:
        return text
    return (
        text[:FUNDAMENTALS_CONTEXT_CHARS]
        + "\n... [fundamental data truncated to fit the Groq token budget]"
    )


def groq_limit_message(detail: str) -> str | None:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None

    message = str(error.get("message", ""))
    code = str(error.get("code", ""))
    if code != "rate_limit_exceeded" and "tokens per minute" not in message.lower():
        return None

    return (
        "Groq rejected this request because it is too large for your current tokens-per-minute "
        "limit. This can happen on the first request because Groq counts the prompt, market "
        "context, attachments, and the requested max output tokens together. Try a shorter "
        "question, remove large attachments, or lower GROQ_MAX_TOKENS/FUNDAMENTALS_CONTEXT_CHARS "
        "in backend/.env."
    )


def market_context_text(state: AgentState) -> str:
    if not state.get("use_market_tools"):
        return "Market tools were not used because this does not look like a market-data question."
    return f"""
Company match:
{state.get("company_match")}

Fundamental data excerpt:
{compact_fundamentals(state.get("fundamentals_data"))}

Yahoo Finance technical data:
{compact_technicals(state.get("technicals"))}
""".strip()


def fallback_answer(state: AgentState) -> str:
    request = state["request"]
    company = state.get("company_match")
    technicals = state.get("technicals")
    if not company and not technicals:
        return (
            "I can answer that, but the model did not return a response. "
            f"Your question was: {request.message}"
        )

    lines = ["Here is the available market context:"]

    if company:
        lines.append(
            f"- Fundamentals: matched {company['company']} and loaded the linked spreadsheet source."
        )

    if technicals and not technicals.get("error"):
        trend = "below" if technicals["ema_20"] < technicals["ema_50"] else "above"
        lines.append(
            f"- Technicals: {technicals['yahoo_symbol']} last close is {technicals['last_close']} "
            f"({technicals['change_pct']}%). RSI(14) is {technicals['rsi_14']}, "
            f"EMA20 is {technicals['ema_20']}, EMA50 is {technicals['ema_50']}, "
            f"MACD is {technicals['macd']} vs signal {technicals['macd_signal']}, "
            f"and ATR(14) is {technicals['atr_14']}. Price is trading with EMA20 {trend} EMA50."
        )
    elif technicals and technicals.get("error"):
        lines.append(f"- Technicals: {technicals['error']}")

    lines.append(
        "- Note: this is analytical context, not financial advice."
    )
    return "\n".join(lines)


def groq_message_content(prompt: str, attachments: list[Attachment]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in attachments:
        if not item.mime_type.startswith("image/"):
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": item.data_url,
                }
            }
        )
    return content


def extract_groq_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return ""


SYSTEM_PROMPT = """
You are a natural, conversational AI assistant, similar in style to ChatGPT.
Answer the user's actual question directly, in a relaxed and helpful voice.
Never reveal hidden reasoning, chain-of-thought, analysis notes, scratchpad text, numbered planning steps, or <think> blocks.
Only return the final answer that should be shown to the user.
Do not use a fixed template, report format, or repeated section headings unless the user's request clearly calls for it.
Let the response shape fit the question: a short paragraph for simple questions, a few bullets for comparisons, and step-by-step detail only when useful.
If the question is not about finance, markets, investing, accounting, or the loaded company, answer normally and ignore the market context.
If the question is finance-related, use the available market data naturally inside the answer. Be practical and clear about uncertainty.
Do not provide guaranteed returns or personalized financial advice. Avoid markdown tables unless the user asks for one.
""".strip()


def call_groq(prompt: str, request: ChatRequest) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "The backend is connected, but GROQ_API_KEY is not set. "
            "Add it to backend/.env and restart the server."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": groq_message_content(f"/no_think\n\n{prompt}", request.attachments),
            }
        ],
        "temperature": 0.7,
        "max_completion_tokens": GROQ_MAX_TOKENS,
        "top_p": 1,
        "stream": False,
    }
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        answer = extract_groq_text(response.json())
        if answer:
            return answer
        raise ValueError("Groq returned no text response.")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        if exc.response is not None and exc.response.status_code in {401, 403}:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Groq rejected the API request. Check that GROQ_API_KEY is active "
                    f"and the selected model '{GROQ_MODEL}' is allowed. "
                    f"Provider response: {detail}"
                ),
            ) from exc
        limit_message = groq_limit_message(detail)
        if limit_message:
            raise HTTPException(status_code=429, detail=limit_message) from exc
        raise HTTPException(status_code=502, detail=f"Groq API error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq call failed: {exc}") from exc


def clean_model_answer(answer: str) -> str:
    draft = re.search(
        r"(?:Draft Construction(?:\s*\([^)]*\))?|Final answer draft)\s*:\s*(.*?)(?=\n\s*\d+\.\s*(?:Check|Review|Validate|Verify)\b|\Z)",
        answer,
        flags=re.I | re.S,
    )
    cleaned = draft.group(1) if draft and re.search(r"<think\b|thinking process|analysis:", answer, flags=re.I) else answer
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"^\s*</think>\s*", "", cleaned, flags=re.I)
    if re.search(r"<think\b[^>]*>", cleaned, flags=re.I):
        final_markers = [
            r"\n\s*(?:Final answer|Final|Answer|Response)\s*:\s*",
            r"\n\s*(?:Short answer|Bottom line)\s*[:\-]\s*",
            r"\n\s*(?:So,|In short,|Based on|Given the data|For Reliance|Reliance\b)",
        ]
        for marker in final_markers:
            match = re.search(marker, cleaned, flags=re.I)
            if match:
                cleaned = cleaned[match.start():]
                break
        else:
            cleaned = re.sub(r"^.*?<think\b[^>]*>.*$", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(
        r"^\s*(?:Here's a thinking process:|Thinking process:|Reasoning:|Analysis:)\s*.*?(?=\n\s*(?:Final answer|Final|Answer|Response|Short answer|Bottom line)\s*[:\-])",
        "",
        cleaned,
        flags=re.I | re.S,
    )
    cleaned = re.sub(r"^\s*(?:Final answer|Final|Answer|Response)\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\n+\s*Sources?:\s*.*(?:\n\s*https?://\S+.*)*$", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"https?://docs\.google\.com/\S+", "[spreadsheet source]", cleaned)
    cleaned = re.sub(r"https?://finance\.yahoo\.com/\S+", "[market data source]", cleaned)
    return cleaned.strip()


def model_node(state: AgentState) -> AgentState:
    request = state["request"]
    recent_history = "\n".join(
        f"{item.role}: {item.content[:HISTORY_MESSAGE_CHARS]}" for item in request.history[-3:]
    )
    image_instruction = (
        "The user attached one or more images. Inspect the attached image(s) directly. "
        "Describe what is visible, identify relevant text/objects/layout/chart elements, "
        "and answer the user's question from the image content. If an image is unclear, say what is unclear."
        if has_image_attachment(request)
        else "No image is attached to this request."
    )
    prompt = f"""
Image handling:
{image_instruction}

Conversation:
{recent_history}

User question:
{request.message}

Market context:
{market_context_text(state)}

Attachments:
{attachment_summary(request.attachments)}

Instructions:
- If images are attached, prioritize what is visible in the images over generic assumptions.
- For finance questions, use the structured fundamental JSON and Yahoo technical data only when relevant.
- Mention only the numbers and context that help answer the user's question.
- Do not infer missing years, rows, tab names, prices, or financial values.
- When discussing a number from fundamentals, name the tab, metric, and period if present.
- If the JSON validation field reports missing tabs or unrecognized tables, mention that limitation briefly only if it matters.
- If a data tool has an error, state the tool error briefly only if it affects the answer.
- Do not print raw source URLs or add a "Sources" section. The app tracks sources separately.
""".strip()
    answer = clean_model_answer(call_groq(prompt, request).strip())
    if not answer:
        answer = fallback_answer(state)
    return {**state, "answer": answer}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("model", model_node)
graph_builder.set_entry_point("tools")
graph_builder.add_edge("tools", "model")
graph_builder.add_edge("model", END)
agent_graph = graph_builder.compile()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": GROQ_MODEL,
        "has_api_key": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/api/companies")
def companies(q: str = "", limit: int = 20) -> list[CompanyMatch]:
    try:
        df = load_fundamentals_index()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load spreadsheet: {exc}") from exc
    if q:
        df = df[df["company"].str.lower().str.contains(q.lower(), na=False)]
    return [
        CompanyMatch(company=row.company, fundamentals_url=row.fundamentals_url)
        for _, row in df.head(limit).iterrows()
    ]


@app.post("/api/chat")
def chat(request: ChatRequest) -> ChatResponse:
    result = agent_graph.invoke({"request": request})
    company = result.get("company_match")
    return ChatResponse(
        answer=result.get("answer", ""),
        company_match=CompanyMatch(**company) if company else None,
        technicals=result.get("technicals"),
        sources=result.get("sources", []),
    )
