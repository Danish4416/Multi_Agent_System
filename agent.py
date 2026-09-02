from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# DUAL MODEL CONFIGURATION
# ==========================================
llm_20b = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1000
).with_retry(stop_after_attempt=5, wait_exponential_jitter=True)

llm_120b = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2000
).with_retry(stop_after_attempt=5, wait_exponential_jitter=True)


# ==========================================
# SEARCH AGENT
# ==========================================
def build_search_agent(topic: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Search Agent. Create an effective search query. Return ONLY the raw query."),
        ("human", "Research Topic:\n{topic}")
    ])
    chain = prompt | llm_120b | StrOutputParser()
    return chain.invoke({"topic": topic})


# ==========================================
# READER AGENT
# ==========================================
def build_reader_agent(search_results: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Reader Agent. Analyze results and pick top 3 URLs. One per line."),
        ("human", "Search Results:\n{search_results}")
    ])
    chain = prompt | llm_20b | StrOutputParser()
    return chain.invoke({"search_results": search_results})


# ==========================================
# WRITER AGENT (Must be exported as writer_chain)
# ==========================================
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Research Writer Agent.
Output ONLY valid Markdown (# for title, ## for sections).
Base facts strictly on provided research context. Never invent data or fake URL markers."""
    ),
    (
        "human",
        """Research Topic: {topic}

Research Material:
{research}

Internal Feedback (if any):
{feedback}

Generate the final report:
# Research Report: {topic}
## Introduction
## Key Findings
## Analysis
## Conclusion
## Sources"""
    )
])

# EXPORT THIS EXACT NAME
writer_chain = writer_prompt | llm_120b | StrOutputParser()


# ==========================================
# CRITIC AGENT (Must be exported as critic_chain)
# ==========================================
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an internal Quality Guard.
If completely valid, reply ONLY with: APPROVED
If invalid, reply: REJECTED: <short bullet points of issues>"""
    ),
    (
        "human", "Draft Report:\n{report}\n\nRaw Research:\n{research}"
    )
])

# EXPORT THIS EXACT NAME
critic_chain = critic_prompt | llm_20b | StrOutputParser()
