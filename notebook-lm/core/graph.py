from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from config import LLM_MODEL, OLLAMA_BASE_URL
import os
from dotenv import load_dotenv

load_dotenv() # This looks for the .env file and loads the keys

# 1. Define the State
class AgentState(TypedDict):
    query: str
    context: str
    response: str
    next_step: str

# 2. Define the Nodes
def classify_input(state: AgentState):
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    # Logic to determine if user wants to search web, save note, or chat
    prompt = f"Classify this query into: 'web_search', 'save_note', or 'chat'. Query: {state['query']}"
    res = llm.invoke(prompt)
    return {"next_step": res.content.strip().lower()}

def web_search_node(state: AgentState):
    # Pass the key directly as a parameter
    search = TavilySearchResults(
        max_results=2, 
        tavily_api_key=os.getenv("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") 
    )
    results = search.invoke(state["query"])
    return {"context": str(results)}

def generate_answer(state: AgentState):
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    full_prompt = f"Context: {state.get('context', 'None')}\n\nQuestion: {state['query']}"
    res = llm.invoke(full_prompt)
    return {"response": res.content}

# 3. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_input)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    lambda x: "web_search" if "web" in x["next_step"] else "generate",
    {"web_search": "web_search", "generate": "generate"}
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()