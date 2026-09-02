# 🔬 Multi-Agent AI Research System

An autonomous research platform built with **Streamlit**, **LangChain**, and **Groq**. It coordinates multiple AI agents to search the web, filter reliable sources, scrape content, and generate a structured research report.

🚀 **Live Demo:** [multiagent-deep-research.streamlit.app](https://multiagent-deep-research.streamlit.app/)

## Features

- Expands vague topics into precise search queries
- Searches the web via the Tavily API
- Filters results down to the top 3 authoritative sources
- Scrapes and cleans page content with BeautifulSoup
- Internal Critic Agent checks the draft for hallucinated facts/dates before finalizing (max 2 revisions)
- Outputs a clean Markdown report: Introduction → Key Findings → Analysis → Conclusion → Sources

## How It Works

1. **Search Agent** (`openai/gpt-oss-120b`) — turns the topic into an optimized search query
2. **Tavily API** — returns web search results
3. **Reader Agent** (`openai/gpt-oss-20b`) — picks the top 3 most reliable URLs
4. **Scraper** (BeautifulSoup) — extracts clean text from each source
5. **Writer Agent** (`openai/gpt-oss-120b`) — synthesizes everything into a Markdown report
6. **Critic Agent** (`openai/gpt-oss-20b`) — checks the draft against the raw research; if rejected, sends it back to the Writer (max 2 loops)

## Why Two Models?

| Agent | Model | Why |
|---|---|---|
| Search | `gpt-oss-120b` | Better at turning vague prompts into strong queries |
| Reader | `gpt-oss-20b` | Simple filtering task, saves token budget |
| Writer | `gpt-oss-120b` | Needs full reasoning power for report quality |
| Critic | `gpt-oss-20b` | Lightweight validation, doesn't need heavy reasoning |

This split keeps report quality high while staying under Groq's free-tier rate limit (8,000 tokens/minute).

## Reliability

- **Retry with backoff:** every LLM call uses `.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)` so temporary `429` rate-limit errors don't crash the app — it waits (2s → 4s → 8s, with random jitter) and retries automatically.
- **Revision cap:** the Writer–Critic loop stops after 2 revisions max, so it can never hang or burn through the rate limit indefinitely.

## Tech Stack

Streamlit · LangChain · Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) · Tavily Search API · BeautifulSoup4 · python-dotenv

## Project Structure

```
├── app.py            # Streamlit UI
├── pipeline.py        # Orchestrates the agent pipeline
├── agent.py            # Agent prompts + LLM setup
├── tools.py             # Tavily search + scraping tools
├── requirements.txt      # Dependencies
├── runtime.txt             # Python version (3.12)
└── .gitignore                # Excludes .env, __pycache__
```

## Run Locally

```bash
git clone https://github.com/Danish4416/Multi_Agent_System.git
cd Multi_Agent_System
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Run it:
```bash
streamlit run app.py
```

## Deployment

Deployed on **Streamlit Community Cloud**. To redeploy:

1. Push to GitHub (never commit `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select this repo, branch `main`, main file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   TAVILY_API_KEY = "your_tavily_api_key_here"
   ```
5. Deploy

🌐 **Live App:** https://multiagent-deep-research.streamlit.app/
