from agent import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)

from tools import (
    web_search,
    scrape_url
)

import re


def run_research_pipeline(
    topic: str,
    progress_callback=None
) -> dict:

    state = {}

    def update(message):

        if progress_callback:
            progress_callback(message)
        else:
            print(message)


    # ==========================================
    # STEP 1 - SEARCH AGENT
    # ==========================================

    update("🔎 Search Agent is planning the research...")

    search_query = build_search_agent(topic)

    state["search_query"] = search_query


    update("🌐 Search Agent is searching the web...")

    search_results = web_search(
        search_query,
        max_results=5
    )

    state["search_results"] = search_results


    # Format search results for Reader Agent
    formatted_results = ""

    for i, result in enumerate(search_results):

        formatted_results += (
            f"\nRESULT {i + 1}\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Snippet: {result.get('content', '')[:500]}\n"
        )


    # ==========================================
    # STEP 2 - READER AGENT
    # ==========================================

    update("📖 Reader Agent is selecting reliable sources...")

    selected_urls_text = build_reader_agent(
        formatted_results
    )


    # Extract URLs selected by Reader Agent
    urls = re.findall(
        r'https?://[^\s]+',
        selected_urls_text
    )


    # Remove duplicates
    urls = list(dict.fromkeys(urls))

    # Maximum 3 sources
    urls = urls[:3]


    # Fallback if Reader Agent doesn't return URLs
    if not urls:

        urls = [
            result.get("url")
            for result in search_results[:3]
            if result.get("url")
        ]


    update("📚 Reader Agent is reading selected sources...")


    scraped_sources = []

    for i, url in enumerate(urls):

        update(
            f"📄 Reading source {i + 1}/{len(urls)}..."
        )

        content = scrape_url(url)

        scraped_sources.append(
            f"""
SOURCE {i + 1}

URL:
{url}

CONTENT:
{content}
"""
        )


    state["scraped_content"] = "\n".join(
        scraped_sources
    )


    # ==========================================
    # COMBINE RESEARCH
    # ==========================================

    research = f"""
SEARCH RESULTS:

{formatted_results[:4000]}


DETAILED SCRAPED SOURCES:

{state["scraped_content"][:10000]}
"""


# ==========================================
    # STEP 3 - WRITER & CRITIC LOOP (HIDDEN)
    # ==========================================

    max_revisions = 2
    internal_feedback = "None"

    for attempt in range(max_revisions + 1):
        
        update("✍️ Writing research report...")

        report = writer_chain.invoke({
            "topic": topic,
            "research": research,
            "feedback": internal_feedback
        })

        update("🧐 Verifying factual accuracy...")

        critique = critic_chain.invoke({
            "report": report,
            "research": research
        })

        # Check if the Critic approved the report
        if "APPROVED" in critique.upper() or attempt == max_revisions:
            state["report"] = report
            break
        else:
            update("🔄 Refining report details...")
            internal_feedback = critique

    # ==========================================
    # COMPLETE
    # ==========================================

    update("✅ Research completed!")

    # Return only the clean report to app.py
    return {"report": state["report"]}