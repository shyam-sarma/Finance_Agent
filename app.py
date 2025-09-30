"""app.py
================
A single-file Flask application implementing a multi-agent financial research workflow
based on the plan described in Agents.md.

Key features
------------
* Declares top-of-file placeholder credentials requested in the spec.
* Defines function tools using the official OpenAI Agents SDK (openai-agents package).
* Builds Discovery, Valuation, Sentiment, and Coordinator agents with detailed
  instructions that describe their responsibilities exhaustively.
* Provides HTTP endpoints to trigger the workflow synchronously using Runner.run_sync.
* Generates PDF reports, evaluates valuation and sentiment signals, and sends email summaries.

Every variable, function, and block includes verbose comments so that a beginner can
follow the logic step by step.
"""

# Standard library imports ------------------------------------------------------
import json  # Allows us to safely serialize structured data for logs and responses.
import logging  # Provides human-readable log messages that help with debugging.
import os  # Gives access to environment variables and filesystem helpers.
import smtplib  # Enables sending emails through an SMTP server such as Gmail.
import ssl  # Supplies TLS/SSL context management for secure SMTP connections.
import time  # Used to add polite pauses between API calls to avoid rate limits.
from datetime import datetime, timedelta  # Supplies precise date calculations for API queries.
from email.message import EmailMessage  # Simplifies creation of MIME email messages with attachments.
from typing import Any, Dict, List, Optional, Tuple  # Adds explicit type hints for clarity.

# Third-party imports -----------------------------------------------------------
from flask import Flask, jsonify, request  # Web framework for the REST API endpoints.
import pandas as pd  # DataFrame computations for handling price histories.
import requests  # HTTP client used for NewsAPI calls.
import yfinance as yf  # Finance data provider for prices and fundamentals.
from reportlab.lib.pagesizes import letter  # Predefined page size for PDF reports.
from reportlab.lib.styles import getSampleStyleSheet  # Provides default PDF text styles.
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # PDF building blocks.

# OpenAI Agents SDK imports (per project instructions) -------------------------
from agents import Agent, Runner, function_tool, ModelSettings  # Official SDK components.

# ------------------------------------------------------------------------------
# Top-of-file placeholder configuration values required by the project spec.
# These are simple string constants that make it trivial to replace with real values later.
OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY"  # Placeholder API key for OpenAI services.
MODEL_NAME: str = "gpt-4.1"  # Default LLM model identifier requested in the plan.
GMAIL_USER: str = "your_email@gmail.com"  # Gmail account that will send summary emails.
GMAIL_APP_PASSWORD: str = "xxxx xxxx xxxx xxxx"  # App-specific password for the Gmail account.
NEWSAPI_KEY: str = "your_newsapi_key"  # Credential used for querying the NewsAPI service.
DEFAULT_UNIVERSE: List[str] = [  # Default ticker list used when the caller omits a universe.
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "NFLX",
    "ORCL",
    "AMD",
]

# Immediately propagate the placeholder OpenAI key into the process environment so that the
# Agents SDK can discover it automatically. This mirrors the behavior of setting the variable
# outside the application but keeps everything inside this prototype file.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure logging with beginner-friendly settings: INFO level and a simple timestamped format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)  # Module-specific logger instance.

# Ensure that the reports directory exists so PDF generation never fails due to missing folders.
REPORTS_DIR: str = os.path.join(os.path.dirname(__file__), "reports")  # Absolute path to reports folder.
os.makedirs(REPORTS_DIR, exist_ok=True)  # Create directory if it does not already exist.

# Flask application setup -------------------------------------------------------
app: Flask = Flask(__name__)  # Primary Flask application object handling HTTP routes.

# Utility helper ----------------------------------------------------------------
def _safe_float(value: Any) -> Optional[float]:
    """Convert arbitrary input into a float when possible.

    Parameters
    ----------
    value: Any
        The value returned by yfinance cashflow DataFrames which might be float, int, or NaN.

    Returns
    -------
    Optional[float]
        A float when conversion succeeds; otherwise ``None`` for missing data.
    """

    # Attempt to cast the input to float while catching exceptions from non-numeric inputs.
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Tool implementations ----------------------------------------------------------
@function_tool()
def batch_download_prices(tickers: List[str], period: str = "1mo", interval: str = "1d") -> str:
    """Download a batch of historical price data using yfinance.

    Parameters
    ----------
    tickers: List[str]
        Symbols to download simultaneously. Keeping the list short helps avoid rate limits.
    period: str, optional
        Date range understood by yfinance (defaults to one month).
    interval: str, optional
        Sampling interval such as daily (default) or weekly.

    Returns
    -------
    str
        Human-readable message describing the download outcome for logging and debugging.
    """

    # Log the tickers for transparency.
    logger.info("batch_download_prices called with tickers=%s", tickers)

    # Request the OHLCV data using yfinance's convenient download function.
    try:
        price_frame: pd.DataFrame = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )  # Combined OHLCV DataFrame potentially using MultiIndex columns.
    except Exception as exc:  # Capture any network or parsing issues.
        logger.exception("Failed to download prices: %s", exc)
        return f"download_failed: {exc}"

    # Persist the data in a simple cache for downstream functions.
    global _PRICE_CACHE
    _PRICE_CACHE: Dict[Tuple[Tuple[str, ...], str, str], pd.DataFrame]  # Cache mapping ticker tuples and parameters to DataFrames.
    if "_PRICE_CACHE" not in globals():  # Initialize cache on first use.
        _PRICE_CACHE = {}
    cache_key: Tuple[Tuple[str, ...], str, str] = (tuple(sorted(tickers)), period, interval)  # Unique identifier for this download.
    _PRICE_CACHE[cache_key] = price_frame  # Store fetched DataFrame for subsequent reuse by discovery logic.

    # Provide a concise confirmation message.
    return "download_success"


@function_tool()
def discover_candidates(tickers: List[str], top_k: int = 5) -> List[str]:
    """Select top-performing tickers from the recent price history.

    Parameters
    ----------
    tickers: List[str]
        Candidate ticker universe supplied by the coordinator agent.
    top_k: int, optional
        Desired number of symbols to return, defaulting to five per specification.

    Returns
    -------
    List[str]
        Sorted list of tickers exhibiting the highest 30-day percentage returns.
    """

    # Ensure at least one ticker is provided to avoid unnecessary API calls.
    if not tickers:
        logger.warning("discover_candidates received an empty ticker list")
        return []

    # Attempt to reuse cached data first; otherwise trigger a fresh download.
    cache_key = (tuple(sorted(tickers)), "1mo", "1d")  # Align with batch_download default parameters.
    price_frame: Optional[pd.DataFrame] = None  # Placeholder for either cached or freshly downloaded prices.
    if "_PRICE_CACHE" in globals() and cache_key in _PRICE_CACHE:
        price_frame = _PRICE_CACHE[cache_key]
    else:
        message: str = batch_download_prices(tickers=tickers)  # Ensure data exists by calling the download helper.
        logger.info("Price download message for discovery: %s", message)
        if message != "download_success":
            return []
        price_frame = _PRICE_CACHE.get(cache_key)

    # Prepare a dictionary to store computed returns keyed by ticker symbol.
    returns: Dict[str, float] = {}  # Maps ticker symbols to their computed 30-day percentage returns.

    # yfinance returns a different shape depending on ticker count; normalize handling.
    for ticker in tickers:
        try:
            # Extract the closing price series for the ticker, accommodating both MultiIndex and single-index outputs.
            if isinstance(price_frame.columns, pd.MultiIndex):
                close_series = price_frame[ticker]["Close"]  # Extract per-ticker close when multi-level columns are present.
            else:
                close_series = price_frame["Close"]  # Single ticker case yields simple column names.

            # Guard against insufficient data.
            if close_series.empty:
                continue

            first_price: float = float(close_series.iloc[0])  # Beginning price for return calculation.
            last_price: float = float(close_series.iloc[-1])  # Most recent closing price.
            if first_price <= 0:
                continue
            pct_return: float = (last_price - first_price) / first_price  # Percentage return over the period.
            returns[ticker] = pct_return
        except Exception as exc:
            logger.exception("Failed to compute return for %s: %s", ticker, exc)
            continue

    # Sort tickers by descending returns and keep the requested count (capped at 10 per spec).
    sorted_tickers: List[str] = [symbol for symbol, _ in sorted(returns.items(), key=lambda item: item[1], reverse=True)]  # Order by descending return.
    limited_tickers: List[str] = sorted_tickers[: min(max(top_k, 5), 10)]  # Enforce 5–10 candidate constraint from requirements.

    logger.info("Discovery selected tickers: %s", limited_tickers)
    return limited_tickers


@function_tool()
def run_quick_dcf(ticker: str) -> Dict[str, float]:
    """Compute a simplified discounted cash flow valuation for a single ticker.

    Parameters
    ----------
    ticker: str
        The symbol for which valuation metrics are computed.

    Returns
    -------
    Dict[str, float]
        Dictionary with current market price, intrinsic fair value, and margin of safety.
    """

    logger.info("run_quick_dcf invoked for %s", ticker)

    # Rate limiting courtesy sleep to respect yfinance policies.
    time.sleep(1.0)  # Pause briefly between valuation calls to behave politely toward the data provider.

    yf_ticker = yf.Ticker(ticker)  # yfinance object encapsulating multiple endpoints for the requested symbol.
    try:
        current_price: float = float(yf_ticker.history(period="1d")["Close"].iloc[-1])  # Latest closing price.
    except Exception as exc:
        logger.exception("Unable to obtain current price for %s: %s", ticker, exc)
        return {"price": 0.0, "fair_value": 0.0, "margin": 0.0}

    cashflow_frame: Optional[pd.DataFrame] = None  # Placeholder for the financial cashflow statement.
    try:
        cashflow_frame = yf_ticker.cashflow
    except Exception as exc:
        logger.exception("Cashflow retrieval failed for %s: %s", ticker, exc)

    free_cash_flow: Optional[float] = None  # Will hold the numeric FCF value if derivable.
    if cashflow_frame is not None and not cashflow_frame.empty:
        if "Free Cash Flow" in cashflow_frame.index:
            free_cash_flow = _safe_float(cashflow_frame.loc["Free Cash Flow"].iloc[0])
        else:
            try:
                cfo = _safe_float(cashflow_frame.loc["Total Cash From Operating Activities"].iloc[0])  # Operating cash flow.
                capex = _safe_float(cashflow_frame.loc["Capital Expenditures"].iloc[0])  # Capital expenditures (typically negative).
                if cfo is not None and capex is not None:
                    free_cash_flow = cfo - capex
            except Exception as exc:
                logger.exception("Fallback FCF calculation failed for %s: %s", ticker, exc)

    if free_cash_flow is None:
        logger.warning("Missing FCF for %s; returning neutral valuation", ticker)
        return {"price": current_price, "fair_value": current_price, "margin": 0.0}

    growth_years: int = 5  # Number of forecast periods using the higher growth rate g1.
    g1: float = 0.08  # Initial five-year growth assumption (8%).
    g2: float = 0.025  # Terminal perpetual growth assumption (2.5%).
    discount_rate: float = 0.10  # Weighted average cost of capital proxy (10%).

    projected_cashflows: List[float] = []  # Stores discounted values for each forecast year.
    for year in range(1, growth_years + 1):
        projected_value: float = free_cash_flow * ((1 + g1) ** year)  # Apply compound growth to FCF.
        discounted_value: float = projected_value / ((1 + discount_rate) ** year)  # Bring future cashflow to present value.
        projected_cashflows.append(discounted_value)

    terminal_cashflow: float = free_cash_flow * ((1 + g1) ** growth_years)  # Final forecast cashflow before terminal stage.
    terminal_value: float = (terminal_cashflow * (1 + g2)) / (discount_rate - g2)  # Gordon growth terminal value formula.
    discounted_terminal_value: float = terminal_value / ((1 + discount_rate) ** growth_years)  # Present value of terminal value.

    fair_value: float = (sum(projected_cashflows) + discounted_terminal_value) / 1_000_000_000  # Very rough normalization placeholder.
    fair_value = max(fair_value, 0.0)  # Prevent negative fair values which would be nonsensical here.

    if fair_value <= 0:
        fair_value = current_price  # Fallback to market price when the simplistic model collapses.

    margin: float = (fair_value - current_price) / current_price if current_price else 0.0  # Margin of safety calculation.

    return {"price": current_price, "fair_value": fair_value, "margin": margin}


@function_tool()
def fetch_sentiment(ticker: str, lookback_days: int = 10) -> Dict[str, Any]:
    """Collect simple sentiment metrics using NewsAPI headlines.

    Parameters
    ----------
    ticker: str
        Symbol whose news sentiment is evaluated.
    lookback_days: int, optional
        How far back to search for articles, defaulting to ten days.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing a normalized sentiment score and list of recent headlines.
    """

    logger.info("fetch_sentiment triggered for %s", ticker)

    end_date: datetime = datetime.utcnow()  # Most recent timestamp used as the query upper bound.
    start_date: datetime = end_date - timedelta(days=lookback_days)  # Lower bound for the search window.

    params: Dict[str, str] = {
        "q": ticker,  # Search keyword.
        "from": start_date.strftime("%Y-%m-%d"),  # Format start date.
        "to": end_date.strftime("%Y-%m-%d"),  # Format end date.
        "sortBy": "publishedAt",  # Prioritize recent articles.
        "language": "en",  # Restrict to English sources.
        "apiKey": NEWSAPI_KEY,  # Authentication token.
        "pageSize": "20",  # Limit number of articles per request.
    }

    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.exception("NewsAPI request failed for %s: %s", ticker, exc)
        return {"score": 0.0, "headlines": [], "warning": str(exc)}

    articles: List[Dict[str, Any]] = payload.get("articles", [])  # Extract article list or default to empty.
    if not articles:
        return {"score": 0.0, "headlines": []}

    positive_keywords: List[str] = ["beat", "surge", "upgrade", "record", "growth", "outperform"]  # Heuristic bullish terms.
    negative_keywords: List[str] = ["miss", "loss", "downgrade", "plunge", "scandal", "lawsuit"]  # Heuristic bearish terms.

    score: float = 0.0  # Running sum of keyword-based scoring.
    headlines: List[str] = []  # Collected headline titles for reporting.

    for article in articles[:5]:  # Limit to top five articles to stay concise.
        title: str = article.get("title") or ""  # Use empty string when title is missing.
        headlines.append(title)
        lowered_title: str = title.lower()  # Lowercase for case-insensitive keyword matching.
        for word in positive_keywords:
            if word in lowered_title:
                score += 1.0
        for word in negative_keywords:
            if word in lowered_title:
                score -= 1.0

    normalized_score: float = max(min(score / 5.0, 1.0), -1.0)  # Normalize to [-1, 1] interval.

    return {"score": normalized_score, "headlines": headlines}


@function_tool()
def generate_pdf(ticker: str, valuation: Dict[str, float], sentiment: Dict[str, Any], decision: str) -> str:
    """Create a one-page PDF summarizing signals for a ticker.

    Parameters
    ----------
    ticker: str
        Symbol included in the PDF filename and report header.
    valuation: Dict[str, float]
        Output from the valuation tool containing price, fair value, and margin.
    sentiment: Dict[str, Any]
        Output from the sentiment tool containing score and headlines.
    decision: str
        Final recommendation string (BUY/HOLD/AVOID).

    Returns
    -------
    str
        Absolute file path of the generated report for downstream emailing.
    """

    report_path: str = os.path.join(REPORTS_DIR, f"{ticker}_report.pdf")  # Destination filepath for the PDF.

    doc = SimpleDocTemplate(report_path, pagesize=letter)  # ReportLab document wrapper using US Letter size.
    styles = getSampleStyleSheet()  # Basic text styles (Title, BodyText, etc.).

    story: List[Any] = []  # Ordered list of Flowables that compose the PDF body.
    story.append(Paragraph(f"<b>{ticker}</b> — Stock Report", styles["Title"]))  # Bold title headline.
    story.append(Spacer(1, 12))  # Add whitespace below the title.

    valuation_section: str = (
        f"<b>Valuation</b><br/>Current Price: ${valuation.get('price', 0):.2f}<br/>"
        f"Fair Value (DCF): ${valuation.get('fair_value', 0):.2f}<br/>"
        f"Margin: {valuation.get('margin', 0) * 100:.2f}%"
    )  # HTML-like string summarizing valuation metrics.
    story.append(Paragraph(valuation_section, styles["BodyText"]))
    story.append(Spacer(1, 12))

    sentiment_section_lines: List[str] = [f"Sentiment Score: {sentiment.get('score', 0):.2f}"]  # Start sentiment bullet list.
    headlines_list: List[str] = sentiment.get("headlines", [])  # Extract top headlines.
    if headlines_list:
        sentiment_section_lines.append("Top Headlines:")
        for headline in headlines_list:
            sentiment_section_lines.append(f"• {headline}")  # Prefix each headline with a bullet for readability.
    sentiment_section: str = "<br/>".join(sentiment_section_lines)  # Join lines using HTML line breaks.
    story.append(Paragraph(f"<b>Sentiment</b><br/>{sentiment_section}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Final Recommendation:</b> {decision}", styles["Heading2"]))  # Conclude with recommendation.

    doc.build(story)  # Render the PDF to disk.

    return report_path  # Provide path so coordinator can attach it to emails.


@function_tool()
def send_email(subject: str, html_body: str, to: str, attachments: List[str]) -> str:
    """Send an email with PDF attachments using Gmail SMTP.

    Parameters
    ----------
    subject: str
        Email subject line summarizing the report.
    html_body: str
        HTML content placed in the message body.
    to: str
        Recipient email address.
    attachments: List[str]
        File paths to include as attachments.

    Returns
    -------
    str
        Status message that indicates success or failure.
    """

    message = EmailMessage()  # Container for the outgoing email message.
    message["From"] = GMAIL_USER  # Sender information.
    message["To"] = to  # Recipient address provided by caller.
    message["Subject"] = subject  # Subject summarizing report contents.
    message.set_content("This email requires an HTML-capable client.")  # Plaintext fallback body.
    message.add_alternative(html_body, subtype="html")  # Preferred HTML body.

    for path in attachments:  # Attach each generated PDF.
        try:
            with open(path, "rb") as file_handle:
                file_data: bytes = file_handle.read()  # Binary content of the PDF.
            message.add_attachment(
                file_data,
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(path),
            )
        except Exception as exc:
            logger.exception("Failed to attach %s: %s", path, exc)

    context = ssl.create_default_context()  # TLS context ensuring secure SMTP connection.
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:  # Connect to Gmail SMTP over SSL.
            smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)  # Authenticate using provided app password.
            smtp.send_message(message)  # Dispatch the email including attachments.
        return "email_sent"
    except Exception as exc:
        logger.exception("Email sending failed: %s", exc)
        return f"email_failed: {exc}"


# Agent configuration ----------------------------------------------------------
DISCOVERY_INSTRUCTIONS: str = (
    "You are the Discovery Agent in a financial research workflow."
    " Your mission is to examine a provided universe of equity tickers, fetch recent price"
    " data using the provided tools, compute 30-day percentage returns, and select between"
    " five and ten tickers with the strongest performance."
    "\n\nFollow these explicit steps:"
    "\n1. Call batch_download_prices with the exact list of tickers you received."
    "\n2. After confirming a successful download, call discover_candidates specifying top_k"
    " equal to the desired number of selections (default five)."
    "\n3. Return only a JSON list of chosen tickers. Do not include prose explanations."
    "\n\nConstraints:"
    "\n- Never fabricate tickers."
    "\n- Always respect the caller's requested top_k bounds of 5–10."
    "\n- If downloads fail, return an empty list to signal the coordinator."
)

VALUATION_INSTRUCTIONS: str = (
    "You are the Valuation Agent tasked with producing a quick discounted cash flow estimate"
    " for a single ticker at a time."
    "\n\nOperating procedure:"
    "\n1. Use run_quick_dcf exactly once per ticker."
    "\n2. Return the dictionary output verbatim as JSON with keys price, fair_value, and margin."
    "\n3. If the tool indicates missing cash flow data, propagate the neutral result without changes."
    "\n\nDo not add commentary or additional calculations beyond the provided tool output."
)

SENTIMENT_INSTRUCTIONS: str = (
    "You are the Sentiment Agent responsible for evaluating recent news sentiment for an"
    " assigned ticker."
    "\n\nProcedure:"
    "\n1. Call fetch_sentiment with the ticker symbol and a reasonable lookback (default 10 days)."
    "\n2. Return the JSON containing score and headlines exactly as produced."
    "\n3. If the tool returns a warning or empty results, forward them without modification."
    "\n\nAvoid any creative writing; remain a structured data provider."
)

COORDINATOR_INSTRUCTIONS: str = (
    "You are the Coordinator Agent orchestrating a comprehensive financial report workflow."
    "\n\nInputs:"
    "\n- A JSON payload describing the ticker universe, desired top_k count, and email recipient."
    "\n\nResponsibilities:"
    "\n1. Invoke the discovery tool to select 5–10 promising tickers."
    "\n2. For each selected ticker, call the valuation and sentiment tools to obtain structured results."
    "\n3. Apply the decision rule: BUY if margin > 0.10 and sentiment.score > 0; HOLD if abs(margin) ≤ 0.10; otherwise AVOID."
    "\n4. Generate a PDF report for each ticker using the PDF tool."
    "\n5. Compose an HTML email summarizing tickers, decisions, prices, and fair values."
    "\n6. Send the email with all PDFs attached using the email tool."
    "\n\nOutput format:"
    "\nReturn a JSON object with keys 'status', 'tickers', 'decisions', 'reports', and 'email_status'."
    "\nIf any ticker fails valuation or sentiment, skip it and add a warning in a field named 'warnings'."
    "\n\nImportant constraints:"
    "\n- Use tools for all external actions; do not guess values."
    "\n- Keep the run within 40 turns."
    "\n- Provide concise JSON-only responses."
)

# Model settings ensure specific tool usage expectations.
REQUIRED_TOOL_SETTINGS: ModelSettings = ModelSettings(tool_choice="required")

# Construct each agent with comprehensive instructions and appropriate tool sets.
discovery_agent: Agent = Agent(
    name="Discovery Agent",
    instructions=DISCOVERY_INSTRUCTIONS,
    model=MODEL_NAME,
    tools=[batch_download_prices, discover_candidates],
    model_settings=REQUIRED_TOOL_SETTINGS,
)

valuation_agent: Agent = Agent(
    name="Valuation Agent",
    instructions=VALUATION_INSTRUCTIONS,
    model=MODEL_NAME,
    tools=[run_quick_dcf],
)

sentiment_agent: Agent = Agent(
    name="Sentiment Agent",
    instructions=SENTIMENT_INSTRUCTIONS,
    model=MODEL_NAME,
    tools=[fetch_sentiment],
)

coordinator_agent: Agent = Agent(
    name="Coordinator Agent",
    instructions=COORDINATOR_INSTRUCTIONS,
    model=MODEL_NAME,
    tools=[
        discovery_agent.as_tool("discover"),
        valuation_agent.as_tool("value"),
        sentiment_agent.as_tool("sentiment"),
        generate_pdf.as_tool("make_report"),
        send_email.as_tool("email"),
    ],
    model_settings=REQUIRED_TOOL_SETTINGS,
)

# Flask routes -----------------------------------------------------------------
@app.get("/health")
def health() -> Any:
    """Simple health check endpoint returning JSON status."""
    return jsonify({"status": "ok"})


@app.post("/run")
def run_workflow() -> Any:
    """Trigger the coordinator agent workflow with user-provided parameters."""

    payload: Dict[str, Any] = request.get_json(force=True) if request.data else {}  # User-supplied JSON body.

    universe: List[str] = payload.get("universe") or DEFAULT_UNIVERSE  # Default to project universe when omitted.
    top_k: int = int(payload.get("top_k", 5))  # Desired number of discovery candidates.
    email_to: str = payload.get("email_to", GMAIL_USER)  # Recipient fallback to sender email for testing.

    input_instructions: str = json.dumps(
        {
            "universe": universe,
            "top_k": top_k,
            "email_to": email_to,
        }
    )  # Coordinator expects a JSON-formatted string describing run parameters.

    try:
        result = Runner.run_sync(
            agent=coordinator_agent,
            input=input_instructions,
            max_turns=40,
        )  # Execute the full workflow synchronously to keep Flask handler simple.
    except Exception as exc:
        logger.exception("Coordinator run failed: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

    final_output: Any = getattr(result, "output", None) or getattr(result, "final_output", None)  # Extract structured agent response.
    if final_output is None:
        final_output = getattr(result, "messages", [])  # Fallback to raw message sequence for debugging scenarios.

    return jsonify({"status": "success", "result": final_output})  # Return consistent success payload.


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
