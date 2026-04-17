import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()
tavily_api_key = os.getenv("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")

llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0,
    timeout=300, 
    keep_alive="1h" 
)


web_search = TavilySearch(max_results=3, api_key=tavily_api_key)

@tool
def summarize(text: str) -> str:
    """Summarize a long piece of text into a short 2-3 sentence paragraph."""
    response = llm.invoke(f"Summarize this concisely:\n\n{text}")
    return response.content

tools = [web_search, summarize]

agent_executor = create_agent(llm, tools=tools)


def run_agent(query: str):
    print(f"\n{'='*20} AGENT STARTING {'='*20}")
    print(f"Task: {query}")
    
    try:
        result = agent_executor.invoke({"input": query})
        
   
        print(f"\n✨ FINAL ANSWER:\n{result.get('output', 'No answer generated.')}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Tip: Ensure you have run 'ollama run llama3.2:3b' in a terminal first.")

if __name__ == "__main__":
    run_agent("Find the latest news on OpenAI and summarize it.")