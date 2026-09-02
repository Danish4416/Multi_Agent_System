from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=3000
)


# ==========================================
# SEARCH AGENT
# ==========================================

def build_search_agent(topic: str):

    """
    Search Agent prepares an optimized search query.
    Tavily performs the actual web search.
    """

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a Search Agent in a Multi-Agent Research System.

Your job is to create an effective web search query
for researching the user's topic.

Focus on:
- Recent information
- Reliable sources
- Official sources
- Important developments
- Facts and statistics

Return ONLY the optimized search query.
"""
        ),

        (
            "human",
            """
Research Topic:

{topic}
"""
        )
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "topic": topic
    })


# ==========================================
# READER AGENT
# ==========================================

def build_reader_agent(search_results: str):

    """
    Reader Agent uses LLM intelligence to select
    the most relevant sources.
    """

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a Reader Agent in a Multi-Agent Research System.

Analyze the provided search results and identify
the most relevant and reliable URLs.

Prioritize:
1. Official government websites
2. Official organization websites
3. Research institutions
4. Reputable news sources

Avoid unreliable sources.

Return ONLY up to 3 URLs.

One URL per line.
Do not include explanations.
"""
        ),

        (
            "human",
            """
Search Results:

{search_results}
"""
        )
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "search_results": search_results
    })


# ==========================================
# WRITER AGENT
# ==========================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert Research Writer Agent. Your objective is to compile research materials into professional, highly accurate reports.

Strict Formatting Rules:
1. Output ONLY standard Markdown text (use # for main headers, ## for section headers).
2. Never invent metadata tags, inline category keys (e.g., #FindingEvidence, [SOURCE 1]), or code wrappers.
3. Ensure natural spacing between all words, dates, and numbers. Do not merge words together.
4. Use standard markdown bolding (**text**) instead of special mathematical or double-asterisk symbols.

Content Rules:
1. Base all facts, statistics, and dates strictly on the provided research context. Never invent data.
2. Clearly distinguish completed events, ongoing trends, and prospective plans.
3. In the Sources section, include ONLY URLs directly provided in the research material.
4. If internal feedback is supplied, address each mentioned revision before finalizing the report.
"""
    ),
    (
        "human",
        """
Research Topic:
{topic}

Research Material:
{research}

Internal Feedback / Revision Notes (if any):
{feedback}

Generate the final research report strictly structured as follows:

# Research Report: {topic}

## Introduction
[Briefly state the context, purpose, and key focus of the research.]

## Key Findings
[Provide a clear list of at least 3 distinct findings with concrete details.]

## Analysis
[Analyze the broader implications, technical/financial impact, and context.]

## Conclusion
[Summarize overall outcomes and potential future outlook.]

## Sources
[List only the valid links from the research material as markdown hyperlinked text.]
"""
    )
])

writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# CRITIC AGENT (Internal Quality Guard)
# ==========================================
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an internal Quality Guard Agent.

Review the research draft against the provided raw research material.

Validation Checks:
1. Ensure no facts, dates, or metrics are fabricated.
2. Ensure structural headers (Introduction, Key Findings, Analysis, Conclusion, Sources) are present.
3. Ensure assertions directly match the raw text.

Output Rules:
- If completely valid, reply ONLY with: APPROVED
- If invalid, reply with: REJECTED: <concise list of fixes needed>
"""
    ),
    (
        "human",
        """
Draft Report:
{report}

Raw Research:
{research}
"""
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()
