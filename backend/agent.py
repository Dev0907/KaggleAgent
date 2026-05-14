import os
import time
import re
from typing import Dict, TypedDict, Any, List, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

class AgentState(TypedDict):
    competition: str  # this is now the Kaggle URL
    model: str
    messages: Sequence[BaseMessage]
    slug: str
    
    # Overview Parser Outputs
    overview_summary: str
    task_type: str
    metric: str
    deadline: str
    prize: str
    
    # Data Analyzer Outputs
    data_description: str
    data_formats: str
    suggested_cv: str
    
    # Discussion Miner Outputs
    key_discussion_points: List[str]
    
    # History Matcher Outputs
    past_winners_strategies: List[str]
    
    # Synthesizer Outputs
    final_strategy: str

def extract_slug(url: str) -> str:
    match = re.search(r'kaggle\.com/competitions/([^/]+)', url)
    if match:
        return match.group(1)
    return url.strip('/')

def search_web(query: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key and api_key != "your_tavily_api_key_here":
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, search_depth="advanced")
            results = "\n".join([res['content'] for res in response.get('results', [])])
            return results
        except Exception as e:
            return f"Search error: {str(e)}"
    else:
        # Mock search if no API key
        time.sleep(2)
        return f"Mocked search results for: {query}"

def call_llm(model_name: str, prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        try:
            llm = ChatGroq(model=model_name, groq_api_key=api_key)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Mocked response (LLM error: {str(e)})"
    else:
        time.sleep(2)
        return f"Mocked LLM generation based on prompt: {prompt[:30]}..."

def create_kaggle_agent():
    workflow = StateGraph(AgentState)
    
    # Agent 1: Overview Parser
    def overview_parser(state: AgentState):
        url = state["competition"]
        slug = extract_slug(url)
        
        # Search web for overview
        search_results = search_web(f"Kaggle {slug} competition overview prize deadline rules")
        
        prompt = f"Based on these search results: {search_results}\n\nSummarize the {slug} Kaggle competition in simple words. Include the prize, deadline, evaluation metric, and task type (e.g. classification, NLP, etc.). Keep it concise and professional."
        summary = call_llm(state.get("model", "llama3-70b-8192"), prompt)
        
        return {
            "slug": slug,
            "overview_summary": summary,
            "messages": [AIMessage(content=f"Analyzed competition overview for {slug}.")]
        }

    # Agent 2: Data Analyzer
    def data_analyzer(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} data description format size variables")
        
        prompt = f"Based on these search results: {search_results}\n\nExplain the dataset for {slug} in simple words. What is the data format? What are the key variables? Suggest a high-level cross-validation strategy."
        data_desc = call_llm(state.get("model", "llama3-70b-8192"), prompt)
        
        return {
            "data_description": data_desc,
            "messages": [AIMessage(content=f"Analyzed data structure for {slug}.")]
        }

    # Agent 3: Discussion Miner
    def discussion_miner(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} discussion insights data issues tips tricks")
        
        prompt = f"Based on these search results: {search_results}\n\nExtract 3-5 key discussion points, issues, or tips for the {slug} competition. Return as a bulleted list."
        points = call_llm(state.get("model", "llama3-70b-8192"), prompt)
        
        return {
            "key_discussion_points": [p.strip() for p in points.split('\n') if p.strip()],
            "messages": [AIMessage(content=f"Mined top discussion signals for {slug}.")]
        }

    # Agent 4: History Matcher
    def history_matcher(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} past winner solutions github writeup similar competitions")
        
        prompt = f"Based on these search results: {search_results}\n\nSummarize the top strategies and approaches used by winners of {slug} or similar past competitions (like previous iterations). Explain clearly what worked best."
        winners_info = call_llm(state.get("model", "llama3-70b-8192"), prompt)
        
        return {
            "past_winners_strategies": [winners_info],
            "messages": [AIMessage(content=f"Researched past winner strategies for {slug}.")]
        }

    # Agent 5: Synthesizer
    def synthesizer(state: AgentState):
        slug = state.get("slug", "competition")
        
        final_strategy = f"Based on all previous analysis, here is the synthesized strategy for {slug}...\n\n"
        final_strategy += "The recommended pipeline involves starting with a strong baseline, focusing heavily on feature engineering, and applying the cross-validation strategy mentioned earlier."
        
        return {
            "final_strategy": final_strategy,
            "messages": [AIMessage(content=f"Synthesized comprehensive final strategy.")]
        }

    workflow.add_node("overview_parser", overview_parser)
    workflow.add_node("data_analyzer", data_analyzer)
    workflow.add_node("discussion_miner", discussion_miner)
    workflow.add_node("history_matcher", history_matcher)
    workflow.add_node("synthesizer", synthesizer)

    workflow.set_entry_point("overview_parser")
    workflow.add_edge("overview_parser", "data_analyzer")
    workflow.add_edge("data_analyzer", "discussion_miner")
    workflow.add_edge("discussion_miner", "history_matcher")
    workflow.add_edge("history_matcher", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
