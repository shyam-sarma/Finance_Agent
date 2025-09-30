Finance Agentic System — Single-File Flask Server (Implementation Plan)

0) Ground rules (so we don’t repeat past mistakes)
	•	We must use OpenAI’s Agents SDK from PyPI: openai-agents.
	•	Docs: OpenAI Agents SDK (Python) guide & quickstart.  ￼
	•	API reference for Agents, Runner, @function_tool.  ￼
	•	Do not write a custom “agent” framework. Import from agents: Agent, Runner, function_tool, ModelSettings. (See examples in the official docs and PyPI.)  ￼
	•	We’ll use Runner.run_sync(...) inside Flask to avoid async boilerplate. It’s an officially supported call.  ￼
	. All agents must have detailed, role-specific instructions prompts (dont worry about the prompt lenght) that clearly define responsibilities, input/output format, and constraints — no vague one-liners. Make the prompts as detailed as possible so that the agents are very effective and will make me a millionaire in 3 months.

⸻

1) Deliverable

A single Python file app.py that:
	•	Exposes POST /run to trigger the workflow.
	•	Builds four agents (Discovery, Valuation, Sentiment, Coordinator).
	•	Uses function tools for: yfinance batch download, discovery filter, DCF calc, NewsAPI sentiment, PDF generation, and Gmail SMTP send.
	•	Emits one-page PDF per stock and emails them.
	•	Reads all credentials from top-of-file placeholders (hardcoded strings for this prototype, as requested).

⸻

2) Environment & versions (pin these)
	•	Python 3.11 or 3.12.
	•	Create a fresh venv and install:

pip install \
  "openai-agents==0.3.2" \
  "openai>=1.50.0" \
  "Flask>=3.0.0" \
  "yfinance>=0.2.50" \
  "pandas>=2.2.2" \
  "requests>=2.32.3" \
  "reportlab>=4.2.0"

Why:
	•	openai-agents==0.3.2 is the current PyPI release (as of Sep 23, 2025).  ￼
	•	Runner.run_sync exists and is documented.  ￼

Sanity checks (must pass):

python -c "import agents, sys; print('agents ok', agents.__version__)"
python -c "from agents import Agent, Runner, function_tool, ModelSettings; print('imports ok')"

If any import fails, you’re not in the right venv or didn’t install openai-agents.

⸻

3) Top-of-file configuration (placeholders)

At the top of app.py, define these constants (strings):
	•	OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
	•	MODEL_NAME = "gpt-4.1"  (override if desired)
	•	GMAIL_USER = "your_email@gmail.com"
	•	GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"  (Google App Password with 2FA; not your normal password)
	•	NEWSAPI_KEY = "your_newsapi_key"
	•	DEFAULT_UNIVERSE = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","ORCL","AMD"]

Note: The SDK expects the OpenAI key in env by default; since you asked for placeholders, we’ll programmatically set it (or set os.environ["OPENAI_API_KEY"]). Docs show standard API key setup in the quickstart.  ￼

⸻

4) Agent roles (instructions your dev should embed)

4.1 Discovery Agent
	•	Purpose: Pick 5–10 candidates from the universe.
	•	Tools it can call:
	•	batch_download_prices(tickers: list[str], period="1mo", interval="1d") -> str
	•	Uses yfinance.download([...]) (batched) to fetch last month of prices (OHLCV).
	•	Returns a short status string; the data stays in memory for the next tool call (developer will pass it through), or the tool can simply re-fetch as needed (simplest approach for now).
	•	discover_candidates(tickers: list[str], top_k: int=5) -> list[str]
	•	Calculates 30-day % return from the downloaded data; picks top-K (default 5).
	•	Instructions:
	•	“Fetch last 30 days of prices (batched). Compute returns. Return a JSON list of top-K tickers (5–10).”

4.2 Valuation Agent
	•	Purpose: Rough Damodaran-style DCF per ticker.
	•	Tool:
	•	run_quick_dcf(ticker: str) -> dict
	•	Pull price, cash flow via yfinance.Ticker(t).cashflow.
	•	Estimate FCF (use “Free Cash Flow” if present; else CFO - CapEx).
	•	Assumptions: 5y growth g1=8%, terminal g2=2.5%, discount wacc=10%.
	•	Returns: { price: float, fair_value: float, margin: float } (margin = (fair−price)/price).

4.3 Sentiment Agent
	•	Purpose: News sentiment (last 7–14 days).
	•	Tool:
	•	fetch_sentiment(ticker: str, lookback_days: int=10) -> dict
	•	Query NewsAPI with q=ticker, English, sorted by recent.
	•	Use a simple keyword heuristic (e.g., +1 for “beat/surge/upgrade”; −1 for “miss/loss/downgrade”).
	•	Return { score: float in [-1..+1], headlines: list[str] (top 3–5) }.

4.4 Coordinator Agent
	•	Purpose: Orchestrate entire run; combine signals; generate PDFs; email them.
	•	Tools available to Coordinator:
	•	discover(...) → exposes Discovery Agent as a tool (agents-as-tools pattern).  ￼
	•	value(...) → exposes Valuation Agent as a tool.
	•	sentiment(...) → exposes Sentiment Agent as a tool.
	•	generate_pdf(ticker: str, valuation: dict, sentiment: dict, decision: str) -> str
	•	Creates single-page PDF with: current price, fair value, margin, sentiment score, top headlines, final recommendation.
	•	Returns absolute file path.
	•	send_email(subject: str, html_body: str, to: str, attachments: list[str]) -> str
	•	SMTP via Gmail (smtp.gmail.com:465, SSL) using GMAIL_USER and GMAIL_APP_PASSWORD.
	•	Decision rule (explicit):
	•	BUY if margin > 0.10 and sentiment.score > 0.
	•	HOLD if abs(margin) ≤ 0.10.
	•	AVOID otherwise.

“Agents as tools” and Manager patterns are first-class in the SDK; we will use sub_agent.as_tool(...) to expose them to the Coordinator.  ￼

⸻

5) Flask API (sync)

5.1 Routes
	•	GET /health → returns {"status":"ok"}
	•	POST /run
Request JSON (optional):

{
  "universe": ["AAPL","MSFT","GOOGL"],
  "top_k": 5,
  "email_to": "your_email@gmail.com"
}

Behavior:
	•	If universe not provided → use DEFAULT_UNIVERSE.
	•	If top_k missing → default 5.
	•	If email_to missing → send to GMAIL_USER.
	•	Coordinator runs the full workflow and returns a JSON summary including chosen tickers, decisions, and the list of generated PDF file paths.

5.2 Using Runner.run_sync(...)
	•	Inside the Flask route we call Runner.run_sync(coordinator_agent, input_message, context=context_obj, max_turns=40).
	•	run_sync is explicitly supported; it runs the agent loop and returns a RunResult.  ￼

⸻

6) Model & settings
	•	Each Agent should define:
	•	name, instructions, model=MODEL_NAME (default gpt-4.1), and tools=[...].
	•	Optionally force tool usage for Discovery/Coordinator steps via ModelSettings(tool_choice="required") when you want to guarantee a tool call. (Docs show ModelSettings.tool_choice usage.)  ￼

⸻

7) Rate-limit hygiene (yfinance & NewsAPI)
	•	Batch price downloads via yfinance.download([...]).
	•	Insert time.sleep(1–2s) between per-ticker valuation/sentiment calls if looping.
	•	If a tool fails due to transient network issues, retry once with a short backoff.
	•	Keep candidate list to 5–10 tickers per run.

⸻

8) Output: one-page PDF per stock

Sections (strict):
	1.	Header: TICKER — Stock Report
	2.	Valuation: Current Price, Fair Value (DCF), Margin (%)
	3.	Sentiment: Score, then 3–5 bullet headlines
	4.	Final Recommendation: BUY/HOLD/AVOID

Filename: reports/{TICKER}_report.pdf (ensure reports/ exists).

⸻

9) Email summary

Subject: Daily Finance Agent Report
HTML Body:
	•	Overview list of all tickers with decision, price, fair value.
Attachments: all generated PDFs.

⸻

10) Tracing & debugging (optional but useful)
	•	Agents SDK auto-traces runs; you can view tool calls and handoffs in the OpenAI dashboard traces viewer (the SDK docs explain tracing & results).  ￼
	•	For dev diagnostics, log each tool’s inputs/outputs briefly (avoid secrets).

⸻

11) Error handling & fallbacks (must have)
	•	yfinance
	•	If cashflow is missing required fields, return a neutral valuation: fair_value = price, margin = 0.0, and mark the decision logic accordingly.
	•	NewsAPI
	•	If zero articles, return score=0 and headlines=[].
	•	Email
	•	If SMTP fails, still return a success JSON with "email_status": "failed" and include the exception message; PDFs are still generated locally.
	•	Coordinator
	•	If a ticker’s valuation or sentiment fails, skip that ticker and continue others, but include a warning in the summary.

⸻

12) Test plan (run these in order)

A. Environment verification
	•	python -c "from agents import Agent, Runner, function_tool; print('agents imports: OK')" (must succeed).  ￼

B. Health check
	•	Start Flask: python app.py → open http://127.0.0.1:5000/health → {"status":"ok"}.

C. Dry run with tiny universe
	•	curl -X POST http://127.0.0.1:5000/run -H "Content-Type: application/json" -d '{"universe":["AAPL","MSFT","GOOGL"],"top_k":2,"email_to":"YOUR_EMAIL"}'
	•	Expect JSON with "status":"success", list of chosen tickers (<=2), decision per ticker, and file paths under reports/. Check your inbox for the email & PDFs.

D. Failure modes
	•	Temporarily break the NewsAPI key → ensure run still completes with score=0 and a warning.
	•	Temporarily block one ticker’s cashflow → ensure that ticker is neutral/hold or skipped, while others complete.

⸻

13) Project skeleton (single file)

app.py (one file) should contain, in this order:
	1.	Imports & os.environ["OPENAI_API_KEY"] set from the placeholder.
	2.	Placeholders (Section 3).
	3.	Flask app initialization and reports/ directory creation.
	4.	All tools defined with @function_tool and type-annotated signatures and clear docstrings.
	•	batch_download_prices
	•	discover_candidates
	•	run_quick_dcf
	•	fetch_sentiment
	•	generate_pdf
	•	send_email
	5.	Agents:
	•	discovery_agent = Agent(..., tools=[batch_download_prices, discover_candidates], model=MODEL_NAME, ...)
	•	valuation_agent = Agent(..., tools=[run_quick_dcf], model=MODEL_NAME, ...)
	•	sentiment_agent = Agent(..., tools=[fetch_sentiment], model=MODEL_NAME, ...)
	•	coordinator_agent = Agent(..., tools=[discovery_agent.as_tool("discover", ...), valuation_agent.as_tool("value", ...), sentiment_agent.as_tool("sentiment", ...), generate_pdf.as_tool("make_report", ...), send_email.as_tool("email", ...)], model=MODEL_NAME, ...)  (agents-as-tools per SDK docs)  ￼
	6.	Flask routes:
	•	GET /health
	•	POST /run → parse JSON → build a single input instruction string embedding the universe & email target, then call:
result = Runner.run_sync(coordinator_agent, input_text, max_turns=40)  ￼
	•	Return a JSON with: chosen tickers, decisions, paths, and email status (as provided by the coordinator’s final output).
	7.	if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=True)

⸻

14) Security notes (for later)
	•	You chose to hardcode secrets for now; when you’re ready, swap to env vars.
	•	If you later containerize, pass secrets via env and never bake them into the image.

⸻

15) Acceptance checklist (what “done” means)
	•	✅ pip show openai-agents prints version 0.3.2 (or newer).  ￼
	•	✅ GET /health returns ok.
	•	✅ POST /run with a 3-ticker universe completes under ~30s, generates PDFs, and sends an email.
	•	✅ If NewsAPI key is wrong, run still completes (score=0 + warning).
	•	✅ If yfinance can’t compute FCF, that ticker is neutral/hold or skipped—server returns success for others.
	•	✅ The code uses from agents import Agent, Runner, function_tool and never a home-rolled agent.  ￼
