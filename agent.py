from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# LLM CONFIGURATION
# ==========================================

# Fast LLM for preliminary tasks to avoid hitting the 8K TPM rate limit
fast_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)

# Writer LLM configured with automatic retries for rate-limit safety
writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
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
            """You are a Search Agent in a Multi-Agent Research System.

Create an effective, high-yield web search query to research the user's topic.

Guidelines:
- Focus on recent information, reliable sources, official announcements, and core facts.
- Do NOT output extra commentary, quotes, or markdown code blocks.
- Return ONLY the raw optimized search query string."""
        ),
        (
            "human",
            "Research Topic:\n{topic}"
        )
    ])

    chain = prompt | fast_llm | StrOutputParser()

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
            """You are a Reader Agent in a Multi-Agent Research System.

Analyze the search results provided and identify up to 3 of the most relevant and authoritative URLs.

Prioritize:
1. Official government or institutional websites
2. Primary official documentation/company pages
3. Established research publications or top tier news outlets

Avoid spam, clickbait, or unreliable domains.

Return ONLY up to 3 URLs, one URL per line. Do not include explanations or numbering."""
        ),
        (
            "human",
            "Search Results:\n{search_results}"
        )
    ])

    chain = prompt | fast_llm | StrOutputParser()

    return chain.invoke({
        "search_results": search_results
    })


# ==========================================
# WRITER AGENT
# ==========================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Research Writer Agent. Your objective is to compile research materials into professional, highly accurate reports.

Strict Formatting Rules:
1. Output ONLY clean Markdown text (use # for main headers, ## for section headers).
2. Never invent metadata tags, inline category keys (e.g., #FindingEvidence, [SOURCE 1]), or raw system labels.
3. Ensure natural spacing between all words, dates, and numbers. Never concatenate text without spaces.
4. Use standard markdown bolding (**text**) instead of special mathematical or double-asterisk characters.

Content Rules:
1. Base all facts, statistics, and dates strictly on the provided research context. Never invent data.
2. Clearly distinguish completed events, ongoing trends, and prospective plans.
3. In the Sources section, include ONLY real URLs directly present in the research material.
4. If internal review feedback is provided, address every requested fix carefully."""
    ),
    (
        "human",
        """Research Topic:
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
[List only the valid links from the research material as markdown hyperlinked text.]"""
    )
])

writer_chain = (
    writer_prompt
    | writer_llm
    | StrOutputParser()
)


# ==========================================
# CRITIC AGENT (Internal Quality Guard)
# ==========================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an internal Quality Guard Agent.

Review the research draft against the provided raw research material.

Validation Checks:
1. Verify that no facts, dates, or metrics are fabricated.
2. Verify that required structural headers (Introduction, Key Findings, Analysis, Conclusion, Sources) exist.
3. Ensure assertions directly match the raw text provided.

Output Rules:
- If completely valid, reply ONLY with: APPROVED
- If invalid, reply with: REJECTED: <concise bullet points of fixes needed>"""
    ),
    (
        "human",
        """Draft Report:
{report}

Raw Research:
{research}"""
    )
])

critic_chain = (
    critic_prompt
    | fast_llm
    | StrOutputParser()
)
