from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
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
You are an expert Research Writer Agent.

Create accurate, detailed and professional
research reports.

Rules:

1. Use ONLY the provided research.
2. Never invent facts.
3. Never invent dates or statistics.
4. Clearly distinguish completed events,
   ongoing activities and future plans.
5. Do not make unsupported claims.
6. Include only URLs actually provided.
"""
    ),

    (
        "human",
        """
Research Topic:

{topic}


Research Material:

{research}


Write a professional research report using:

# Introduction

# Key Findings
Explain at least 3 important findings.

# Analysis
Analyze implications and importance.

# Conclusion

# Sources
List only actual URLs from the research.
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