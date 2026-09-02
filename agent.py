from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# DUAL MODEL CONFIGURATION
# ==========================================

# 20B Model: Lightweight, lower latency for operational agents
llm_20b = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)

# 120B Model: High reasoning & deep synthesis for the Writer Agent
llm_120b = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)


# ==========================================
# SEARCH AGENT (Uses GPT-OSS-20B)
# ==========================================

def build_search_agent(topic: str):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Search Agent in a Multi-Agent System.
Generate a targeted, high-yield web search query for the user's research topic.
Return ONLY the raw search query string without quotes or extra text."""
        ),
        ("human", "Research Topic:\n{topic}")
    ])

    chain = prompt | llm_20b | StrOutputParser()
    return chain.invoke({"topic": topic})


# ==========================================
# READER AGENT (Uses GPT-OSS-20B)
# ==========================================

def build_reader_agent(search_results: str):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Reader Agent in a Multi-Agent System.
Analyze the search results and pick up to 3 of the most reliable and relevant URLs.

Rules:
- Prioritize official sources, research bodies, and major news outlets.
- Return ONLY up to 3 URLs, one URL per line.
- Do NOT include explanations, numbers, or bullet points."""
        ),
        ("human", "Search Results:\n{search_results}")
    ])

    chain = prompt | llm_20b | StrOutputParser()
    return chain.invoke({"search_results": search_results})


# ==========================================
# WRITER AGENT (Uses GPT-OSS-120B)
# ==========================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Research Writer Agent. Your job is to create detailed, factual research reports.

Formatting Rules:
1. Output ONLY valid Markdown (# for main title, ## for sections).
2. Ensure clear spacing between all words and numbers (never squish text together).
3. Do NOT invent internal metadata tags (e.g., #FindingEvidence, [SOURCE 1]).

Content Rules:
1. Rely ONLY on the provided research context. Never fabricate facts or dates.
2. Under Sources, list only real URLs provided in the research context.
3. Address any internal reviewer feedback provided."""
    ),
    (
        "human",
        """Research Topic:
{topic}

Research Material:
{research}

Internal Feedback (if any):
{feedback}

Generate the final report using this structure:

# Research Report: {topic}

## Introduction
## Key Findings
## Analysis
## Conclusion
## Sources"""
    )
])

writer_chain = writer_prompt | llm_120b | StrOutputParser()


# ==========================================
# CRITIC AGENT (Uses GPT-OSS-20B)
# ==========================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an internal Quality Guard Agent.

Check the report draft against the raw research text.
1. Ensure no facts or numbers are hallucinated.
2. Confirm essential Markdown headers are present.

Output Rules:
- If valid, reply strictly: APPROVED
- If invalid, reply: REJECTED: <short bullet points of issues>"""
    ),
    (
        "human",
        "Draft Report:\n{report}\n\nRaw Research:\n{research}"
    )
])

critic_chain = critic_prompt | llm_20b | StrOutputParser()
