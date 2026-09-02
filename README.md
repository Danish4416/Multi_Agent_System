# 🔬 Multi-Agent AI Research System

An autonomous, production-ready multi-agent research platform built using **Streamlit**, **LangChain**, and **Groq LPUs**. The application orchestrates specialized AI agents to expand user intent, retrieve primary web sources, select authoritative links, extract live page content, and generate publication-grade Markdown research reports backed by an internal quality-guard validation loop.

## 🚀 Live Demo

👉 **[Try the Multi-Agent AI Research System](https://multiagent-deep-research.streamlit.app/)**

## Features

- 🧠 **Query Intent Expansion** – Transforms vague search terms (e.g. "spiderman") into precise, high-yield research queries
- 🌐 **Live Web Search** – Uses the Tavily Search API to retrieve up-to-date, relevant results
- 🔗 **Authoritative Source Selection** – Filters results to prioritize official, research, and top-tier news sources
- 📚 **Automated Web Scraping** – Extracts clean text content from selected URLs using BeautifulSoup
- 🛡 **Self-Correcting Verification** – An internal Critic Agent cross-checks the draft report against raw research to catch hallucinations, fabricated dates, and structural gaps
- 🔄 **Bounded Revision Loop** – The Writer–Critic feedback loop is capped at 2 revisions to guarantee predictable runtime and avoid rate-limit exhaustion
- 🖥 **Clean UX** – Internal revision cycles are hidden from the UI; only high-level progress and the final report are shown
- 📝 **Structured Markdown Output** – Reports always follow Introduction → Key Findings → Analysis → Conclusion → Sources

## System Architecture & Workflow

The platform uses a **Dual-Model Strategy** to balance high-reasoning output quality against Groq's free-tier rate limits (8,000 Tokens Per Minute).

```
 ┌────────────────────────┐
 │     User Topic Input    │
 └───────────┬────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│  🔎 Search Planning Agent (openai/gpt-oss-120b)             │
│  • Expands raw input into a precise, high-yield search      │
│    query                                                     │
└───────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────┐
│  🌐 Tavily Web Search API                                    │
│  • Fetches primary search results and page context          │
└───────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────┐
│  📖 Reader / Filter Agent (openai/gpt-oss-20b)               │
│  • Evaluates results for authority and relevance             │
│  • Selects the top 3 trusted URLs                            │
└───────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────┐
│  📚 Scraper (BeautifulSoup)                                  │
│  • Extracts clean text content from selected sources         │
└───────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────┐
│  ✍️ Writer Agent (openai/gpt-oss-120b)                       │
│  • Synthesizes scraped content into a structured Markdown    │
│    report                                                     │
└───────────────────────────────┬─────────────────────────────┘
                                 │  ◄────────────────┐
                                 ▼                    │ Rejected
┌────────────────────────────────────────────────────┴────────┐
│  🧐 Critic / Quality Guard Agent (openai/gpt-oss-20b)         │
│  • Verifies facts, dates, and figures against raw research    │
│  • Confirms required section headers are present              │
│  • Max 2 revision loops                                        │
└───────────────────────────────┬───────────────────────────────┘
                                 │ Approved / Max Loop Reached
                                 ▼
                     ┌────────────────────────┐
                     │   Final Markdown Report │
                     └────────────────────────┘
```

## Dual-Model Workload Allocation

| Agent | Model | Role & Justification |
|---|---|---|
| **Search Agent** | `openai/gpt-oss-120b` | Query planning — expands ambiguous prompts into high-yield search terms |
| **Reader Agent** | `openai/gpt-oss-20b` | URL filtering — fast, low-latency selection of the top 3 trusted sources |
| **Writer Agent** | `openai/gpt-oss-120b` | Report synthesis — full reasoning power for structured, accurate reports |
| **Critic Agent** | `openai/gpt-oss-20b` | Fact verification — checks the draft without draining high-tier token quota |

## API & System Resilience

**Rate-Limit Protection (Exponential Backoff + Jitter)**

To handle Groq's `429 Too Many Requests` errors on the free tier, every model call is wrapped with LangChain's retry handler (built on `tenacity`):

```python
llm = ChatGroq(...).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)
```

- **Exponential backoff** progressively doubles the wait time between retries (e.g. 2s → 4s → 8s), giving the rate-limit window time to reset.
- **Jitter** adds random variance to each wait interval, preventing multiple agent calls from retrying in lockstep and re-triggering the limit ("thundering herd" effect).

**Bounded Revision Loop**

The Writer–Critic feedback loop enforces a hard cap of `max_revisions = 2`. If the Critic hasn't approved the draft after two revisions, the pipeline finalizes the latest version automatically — preventing infinite loops and runaway token usage.

## Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain (Core & Community)
- **Inference:** Groq LPU Acceleration — `openai/gpt-oss-120b` & `openai/gpt-oss-20b`
- **Web Search:** Tavily Search API
- **Scraping & Parsing:** BeautifulSoup4, Requests
- **Config & Validation:** python-dotenv, pydantic

## Project Structure

```
├── app.py              # Streamlit UI and page logic
├── pipeline.py          # Multi-agent orchestration loop
├── agent.py              # Agent prompts, LLM setup, and retry logic
├── tools.py               # Tavily search + BeautifulSoup scraping utilities
├── requirements.txt        # Production dependencies
├── runtime.txt               # Python version pin (3.12)
├── .gitignore                  # Excludes .env, __pycache__, .vscode
└── README.md                     # Project documentation
```

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Danish4416/Multi_Agent_System.git
cd Multi_Agent_System
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

## Deployment

Deployed on **Streamlit Community Cloud**:

1. Push the repository to GitHub (excluding `.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Create a new app, selecting this repo, the `main` branch, and `app.py` as the entry point.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   TAVILY_API_KEY = "your_tavily_api_key_here"
   ```
5. Click **Deploy**.

🌐 **Live App:** [https://multiagent-deep-research.streamlit.app/](https://multiagent-deep-research.streamlit.app/)

## License

This project is open source and available for personal and educational use.
