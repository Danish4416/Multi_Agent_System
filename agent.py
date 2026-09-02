from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# DUAL MODEL CONFIGURATION
# ==========================================

# 20B Model: Lightweight tasks (Reader & Critic) to conserve rate limits
llm_20b = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)

# 120B Model: Reserved for key cognitive steps (Search Planning & Writer Synthesis)
llm_120b = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2000
).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True
)

# ==========================================
# SEARCH AGENT (Now upgraded to 120B)
# ==========================================
def build_search_agent(topic: str):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Search Agent in a Multi-Agent Research System.
Create an effective, high-yield web search query for the user's research topic.
Return ONLY the raw search query string without quotes or extra text."""
        ),
        ("human", "Research Topic:\n{topic}")
    ])

    chain = prompt | llm_120b | StrOutputParser()
    return chain.invoke({"topic": topic})

# ==========================================
# READER AGENT (Kept on 20B)
# ==========================================
def build_reader_agent(search_results: str):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Reader Agent in a Multi-Agent System.
Analyze the search results and pick up to 3 of the most reliable and relevant URLs.
Return ONLY up to 3 URLs, one URL per line without explanations or numbers."""
        ),
        ("human", "Search Results:\n{search_results}")
    ])

    chain = prompt | llm_20b | StrOutputParser()
    return chain.invoke({"search_results": search_results})

# ==========================================
# WRITER AGENT (Kept on 120B)
# ==========================================
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Research Writer Agent. Compile research materials into professional reports.

Rules:
1. Output ONLY valid Markdown (# for title, ## for sections).
2. Ensure clear spacing between all words and numbers.
3. Base facts strictly on provided research context. Never invent data.
4. Under Sources, list only real URLs provided in the research material."""
    ),
    (
        "human",
        """Research Topic: {topic}

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
# CRITIC AGENT (Kept on 20B)
# ==========================================
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an internal Quality Guard Agent.
Review the research draft against raw research text.
If completely valid, reply ONLY with: APPROVED
If invalid, reply: REJECTED: <short bullet points of issues>"""
    ),
    (
        "human", "Draft Report:\n{report}\n\nRaw Research:\n{research}"
    )
])

critic_chain = critic_prompt | llm_20b | StrOutputParser()
