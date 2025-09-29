
📌 Finance Agentic System – Implementation Plan

🎯 Goal

Build a single-file Flask server that runs a daily stock analysis workflow using OpenAI Agents SDK. The workflow includes discovery, valuation (DCF), sentiment analysis, report generation, and emailing results. It should run on any machine with minimal setup and allow credentials/keys to be easily plugged in.

⸻

1. Architecture Overview
	•	Framework: Python + Flask
	•	Agent orchestration: OpenAI Agents SDK
	•	Endpoints:
	•	/run → triggers the full workflow manually
	•	(optionally) /health → returns "ok" for status checks
	•	Workflow Roles:
	1.	Discovery Agent – selects 5–10 promising stocks from a universe (default: S&P 500 subset).
	2.	Valuation Agent – runs a quick Damodaran-style DCF model.
	3.	Sentiment Agent – pulls recent news via NewsAPI and assigns a sentiment score.
	4.	Coordinator Agent – combines results, generates PDF reports, and emails them.

⸻

2. Data Sources
	•	yfinance → for stock prices + financial statements
	•	NewsAPI → for news & headlines (sentiment)
	•	No DuckDB → keep simple; fetch on the fly
	•	PDF output → ReportLab

⸻

3. Workflow Steps
	1.	Discovery
	•	Input: stock universe (list of tickers, default hardcoded)
	•	Logic:
	•	Pull last 30 days of price data (batch call via yfinance)
	•	Calculate % change over the window
	•	Select top 5–10 performers
	2.	Valuation
	•	Input: ticker
	•	Logic:
	•	Pull cash flow statement, estimate Free Cash Flow
	•	Assume growth (e.g., 8% for 5 years, 2.5% terminal, 10% discount)
	•	Compute intrinsic value per share
	•	Compare to market price → margin of safety
	3.	Sentiment
	•	Input: ticker
	•	Logic:
	•	Query NewsAPI for ticker/company name (last 7–14 days)
	•	Parse headlines, apply keyword-based sentiment scoring
	•	Return score (-1 to +1) and top 3 headlines
	4.	Coordinator
	•	Input: Discovery results + valuations + sentiment
	•	Logic:
	•	Apply rule-based decision:
	•	BUY if undervalued (>10%) and sentiment positive
	•	HOLD if fairly valued (±10%)
	•	AVOID otherwise
	•	Generate a one-page PDF per stock
	•	Send summary email with PDFs attached

⸻

4. Flask Server Structure
	•	Single file: app.py
	•	Contains:
	•	Imports + credentials placeholders
	•	Tool functions (discovery, valuation, sentiment, PDF, email)
	•	Agent definitions (Discovery, Valuation, Sentiment, Coordinator)
	•	Flask routes
	•	Runner call (OpenAI Agents SDK)

⸻

5. Credentials & Placeholders

At the top of app.py, include placeholders:

OPENAI_API_KEY   = "YOUR_OPENAI_API_KEY"
GMAIL_USER       = "your_email@gmail.com"
GMAIL_PASS       = "your_gmail_app_password"   # 2FA → App Password
NEWSAPI_KEY      = "your_newsapi_key"
DEFAULT_UNIVERSE = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","ORCL","AMD"]


⸻

6. OpenAI Agents SDK Usage
	•	Wrap each role as an Agent with clear instructions.
	•	Expose function tools (@function_tool) for:
	•	Downloading prices
	•	Discovering candidates
	•	Running DCF
	•	Fetching sentiment
	•	Generating PDFs
	•	Sending email
	•	Coordinator Agent orchestrates the sub-agents as tools.

⸻

7. Example Request Flow
	1.	Call: POST /run
Body: optional JSON → { "universe": ["AAPL","MSFT"], "email_to": "me@gmail.com" }
	2.	System:
	•	Discovery agent → selects 5 tickers
	•	For each: valuation agent + sentiment agent
	•	Coordinator agent → applies rules, generates PDFs
	•	Email tool → sends summary + attachments
	3.	Response:
JSON with run summary: { "status": "success", "reports": ["AAPL_report.pdf", ...] }

⸻

8. Output
	•	Reports: /reports/{ticker}_report.pdf
	•	Email: Subject = "Daily Finance Agent Report", Body = summary list, Attachments = PDFs
	•	Flask response: run status + file paths

⸻

9. Deployment
	•	Local run: python app.py → Flask starts at http://127.0.0.1:5000
	•	Server run: same file can run on any Linux/Windows machine with Python 3.11+
	•	Docker (optional): add Dockerfile for containerized use later

