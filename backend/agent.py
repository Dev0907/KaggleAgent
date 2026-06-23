import os
import time
import re
import threading
from typing import Dict, TypedDict, Any, List, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

import contextvars

# Context variables to track the current node and keys during execution
current_node_context = contextvars.ContextVar("current_node", default="default")

# Parse multiple API keys from environment
def get_groq_api_keys():
    keys_str = os.environ.get("GROQ_API_KEY", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    return []

def get_tavily_api_keys():
    keys_str = os.environ.get("TAVILY_API_KEY", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    return []

GROQ_API_KEYS = get_groq_api_keys()
TAVILY_API_KEYS = get_tavily_api_keys()

_groq_key_index = 0
_tavily_key_index = 0
_key_lock = threading.Lock()
_tavily_lock = threading.Lock()

def get_next_groq_key():
    global _groq_key_index
    if not GROQ_API_KEYS:
        return None
    with _key_lock:
        key = GROQ_API_KEYS[_groq_key_index % len(GROQ_API_KEYS)]
        _groq_key_index += 1
        return key

def get_next_tavily_key():
    global _tavily_key_index
    if not TAVILY_API_KEYS:
        return None
    with _tavily_lock:
        key = TAVILY_API_KEYS[_tavily_key_index % len(TAVILY_API_KEYS)]
        _tavily_key_index += 1
        return key

class AgentState(TypedDict):
    competition: str  # this is now the Kaggle URL
    messages: Sequence[BaseMessage]
    slug: str
    
    explanation: str
    data: str
    approaches: str
    winners: str
    discussion: str
    external_links: str

def extract_slug(url: str) -> str:
    match = re.search(r'kaggle\.com/competitions/([^/]+)', url)
    if match:
        return match.group(1)
    return url.strip('/')

@tool
def search_web(query: str) -> str:
    """Searches the web for the given query using Tavily and returns the results as a string."""
    node_name = current_node_context.get()
    tavily_keys = get_tavily_api_keys()
    
    # Try all Tavily keys on failure, starting with the designated one for this node
    keys_to_try = []
    if tavily_keys:
        node_to_index = {
            "explanation": 0,
            "data": 1,
            "approaches": 2,
            "winners": 3,
            "discussion": 4,
            "code_hunter": 5,
            "chat": 0
        }
        if node_name in node_to_index:
            start_idx = node_to_index[node_name] % len(tavily_keys)
            keys_to_try = [tavily_keys[(start_idx + i) % len(tavily_keys)] for i in range(len(tavily_keys))]
        else:
            keys_to_try = tavily_keys
    else:
        env_key = os.environ.get("TAVILY_API_KEY")
        if env_key:
            keys_to_try = [env_key]
            
    if not keys_to_try or all(k == "your_tavily_api_key_here" for k in keys_to_try):
        return f"Mocked search results for: {query}"
        
    last_error = None
    for api_key in keys_to_try:
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, search_depth="advanced")
            results = "\n".join([res['content'] for res in response.get('results', [])])
            if results:
                return results
        except Exception as e:
            last_error = e
            print(f"Tavily search error with key {api_key[:10]}...: {str(e)}")
            
    return f"Search error (tried all keys): {str(last_error)}"

def call_llm(prompt: str, system_message: str = "You are a Kaggle Grandmaster and Search Expert.", node_name: str = None) -> str:
    """Calls the GPT-OSS 120B model (with Llama-3 fallback) with tool-calling for end-to-end search and reasoning."""
    # Set the context for the current node name
    token = None
    if node_name:
        token = current_node_context.set(node_name)
        
    try:
        groq_keys = get_groq_api_keys()
        
        # Determine starting index for the key
        node_to_index = {
            "explanation": 0,
            "data": 1,
            "approaches": 2,
            "winners": 3,
            "discussion": 4,
            "code_hunter": 5,
            "chat": 0
        }
        
        start_idx = 0
        if groq_keys:
            if node_name in node_to_index:
                start_idx = node_to_index[node_name] % len(groq_keys)
                
        # Models to try sequentially on failure (allows fallback from custom to standard Groq models)
        models_to_try = ["openai/gpt-oss-120b", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]
        
        if not groq_keys or all(k == "your_groq_api_key_here" for k in groq_keys):
            return f"Mocked LLM generation for: {prompt[:50]}..."
            
        last_error = None
        for model in models_to_try:
            # For each model, try all available API keys
            for attempt in range(len(groq_keys)):
                key_idx = (start_idx + attempt) % len(groq_keys)
                api_key = groq_keys[key_idx]
                
                try:
                    llm = ChatGroq(model=model, groq_api_key=api_key)
                    tools = [search_web]
                    llm_with_tools = llm.bind_tools(tools)
                    
                    messages = [
                        ("system", system_message),
                        ("human", prompt)
                    ]
                    
                    # Initial call
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Loop to handle tool calls (Search -> Reason -> Search)
                    # Limiting to 3 tool calls to prevent infinite loops
                    for _ in range(3):
                        if not response.tool_calls:
                            break
                            
                        for tool_call in response.tool_calls:
                            if tool_call["name"] == "search_web":
                                tool_output = search_web.invoke(tool_call["args"])
                                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                        
                        # Get the final synthesis or next tool call
                        response = llm_with_tools.invoke(messages)
                        messages.append(response)
                        
                    return response.content
                except Exception as e:
                    last_error = e
                    print(f"Error calling model {model} with key index {key_idx}: {str(e)}")
                    # Small backoff before retrying
                    time.sleep(0.5)
                    
        return f"End-to-End Model Error (Tried all fallback models and keys): {str(last_error)}"
    finally:
        if token:
            current_node_context.reset(token)

def create_kaggle_agent():
    workflow = StateGraph(AgentState)
    
    # Node 1: Universal Explanation (The Grandmaster Overview)
    def explanation_node(state: AgentState):
        url = state["competition"]
        slug = extract_slug(url)
        
        prompt = f"""
        Act as a Search-First Kaggle Grandmaster. 
        Your task is to provide a Strategic Overview for: {url} (Slug: {slug})
        
        INSTRUCTIONS:
        1. Use your search tool to find the competition mission, historical context, and technical hurdles.
        2. Synthesize the findings into:
           - THE MISSION: Predictive task and value.
           - CRITICAL SYSTEM PROBLEMS: Hurdles like data shifts, noise, or constraints.
           - DOMAIN ARCHETYPE: Time-Series, CV, or Tabular?
        """
        summary = call_llm(prompt, node_name="explanation")
        
        return {
            "slug": slug,
            "explanation": summary
        }

    # Node 2: Universal Data Dissection (Deep Architectural Audit)
    def data_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Act as an Elite Data Architect.
        Perform a Deep Data Audit for Kaggle competition: {slug}
        
        INSTRUCTIONS:
        1. Search for the official data descriptions, target variables, and metrics.
        2. Audit the:
           - DATA STRUCTURE: File hierarchy.
           - TARGET ANALYSIS: Regression vs Classification.
           - THE EVALUATION: Deep-dive into the specific metric (Sharpe, Correlation, etc.)
           - DATA CHALLENGES: Leaks, temporal constraints.
        """
        data_desc = call_llm(prompt, node_name="data")
        
        return {
            "data": data_desc
        }

    # Node 3: Elite Approaches (The Top 3 & Feasible Entry)
    def approaches_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Act as a Grandmaster ML Architect.
        Design 3 "Elite" Approaches for competition: {slug}
        
        INSTRUCTIONS:
        1. Search for high-voted code, top notebooks, and baseline strategies.
        2. Detail 3 strategies: Name, Architecture (models), Rationale, and unique Feature Engineering ideas.
        3. Provide the "Universal Most Feasible Approach" for a fast-start submission.
        """
        approaches = call_llm(prompt, node_name="approaches")
        
        return {
            "approaches": approaches
        }

    # Node 4: Grandmaster Secret Sauce (The Winning Edge)
    def winners_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Act as a Competitive Intelligence Expert.
        Deep-Dive into the TOP 5 WINNING SOLUTIONS for: {slug}
        
        INSTRUCTIONS:
        1. Search for technical writeups, gold medal solutions, and winning team blogs.
        2. Detail: Rank/User, Core Idea, Technical Stack, CV vs LB stability, and the "Secret Sauce".
        """
        winners_info = call_llm(prompt, node_name="winners")
        
        return {
            "winners": winners_info
        }

    # Node 5: Universal Forum Intel (The Pulse of the Crowd)
    def discussion_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Act as a Community Intel Specialist.
        Brief us on the "Hot Intelligence" for: {slug}
        
        INSTRUCTIONS:
        1. Search Kaggle forums for upvoted discussions, leak alerts, and Golden Kernels.
        2. Identify: Top-voted buzz, elite forum strategies, alerts/leaks, and community-endorsed workflows.
        """
        discussion_info = call_llm(prompt, node_name="discussion")
        
        return {
            "discussion": discussion_info
        }


    # Node 6: External Code Hunter (GitHub & Article Scout)
    def code_hunter_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Act as an External Resource Scout.
        Find ELITE external resources for: {slug}
        
        INSTRUCTIONS:
        1. Search GitHub for winning code, Medium/blog writeups, and relevant Research Papers.
        2. Provide a linked list with 1-sentence summaries for each.
        """
        links_info = call_llm(prompt, node_name="code_hunter")
        
        return {
            "external_links": links_info
        }

    workflow.add_node("explanation", explanation_node)
    workflow.add_node("data", data_node)
    workflow.add_node("approaches", approaches_node)
    workflow.add_node("winners", winners_node)
    workflow.add_node("discussion", discussion_node)
    workflow.add_node("code_hunter", code_hunter_node)

    workflow.set_entry_point("explanation")
    workflow.add_edge("explanation", "data")
    workflow.add_edge("data", "approaches")
    workflow.add_edge("approaches", "winners")
    workflow.add_edge("winners", "discussion")
    workflow.add_edge("discussion", "code_hunter")
    workflow.add_edge("code_hunter", END)

    return workflow.compile()
