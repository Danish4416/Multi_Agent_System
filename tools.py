import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================================
# SEARCH TOOL
# ==========================================

def web_search(query: str, max_results: int = 5) -> list:

    try:
        response = tavily.search(
            query=query,
            max_results=max_results
        )

        return response.get("results", [])

    except Exception as e:
        print(f"Search Error: {e}")
        return []


# ==========================================
# SCRAPING TOOL
# ==========================================

def scrape_url(url: str) -> str:

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        return text[:4000]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"