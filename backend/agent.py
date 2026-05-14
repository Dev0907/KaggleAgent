import os
import time
import re
from typing import Dict, TypedDict, Any, List, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

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
        return f"Mocked search results for: {query}"

def call_llm(prompt: str, system_message: str = "You are a Kaggle Grandmaster and Search Expert.") -> str:
    """Calls the GPT-OSS 120B model with tool-calling for end-to-end search and reasoning."""
    model_name = "openai/gpt-oss-120b"
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key or api_key == "your_groq_api_key_here":
        return f"Mocked LLM generation for: {prompt[:50]}..."

    try:
        llm = ChatGroq(model=model_name, groq_api_key=api_key)
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
        return f"End-to-End Model Error (GPT-OSS 120B): {str(e)}"

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
        summary = call_llm(prompt)
        
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
        data_desc = call_llm(prompt)
        
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
        approaches = call_llm(prompt)
        
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
        winners_info = call_llm(prompt)
        
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
        discussion_info = call_llm(prompt)
        
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
        links_info = call_llm(prompt)
        
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
