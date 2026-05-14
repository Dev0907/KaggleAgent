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
    messages: Sequence[BaseMessage]
    slug: str
    
    explanation: str
    data: str
    approaches: str
    winners: str
    discussion: str

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

def call_llm(prompt: str) -> str:
    # The user asked for a large open-source model available via Groq.
    # llama-3.3-70b-versatile is currently the top tier on Groq.
    model_name = "llama-3.3-70b-versatile"
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        try:
            llm = ChatGroq(model=model_name, groq_api_key=api_key)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            # Fallback to a simpler model if 3.3 is offline
            try:
                llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key)
                response = llm.invoke(prompt)
                return response.content
            except Exception as e2:
                return f"Mocked response (LLM error: {str(e2)})"
    else:
        time.sleep(2)
        return f"Mocked LLM generation based on prompt: {prompt[:30]}..."

def create_kaggle_agent():
    workflow = StateGraph(AgentState)
    
    # Node 1: Universal Explanation
    def explanation_node(state: AgentState):
        url = state["competition"]
        slug = extract_slug(url)
        
        # Deep search across multiple vectors
        search_results = search_web(f"Kaggle competition {slug} overview task goal objective problem statement")
        
        prompt = f"""
        Competition: {slug}
        URL: {url}
        Search Context: {search_results}
        
        Task: Provide an Elite Universal Analysis of this Kaggle competition.
        1. Core Mission: What is the primary problem being solved?
        2. Impact: Why does this competition exist? (Scientific, commercial, or social value).
        3. Challenge Level: How difficult is this for a beginner vs. an expert?
        
        Explain in a high-octane, engaging, and clear manner. Use Grandmaster-level terminology but keep it accessible.
        """
        summary = call_llm(prompt)
        
        return {
            "slug": slug,
            "explanation": summary
        }

    # Node 2: Universal Data Dissection
    def data_node(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} dataset EDA features target leak validation strategy")
        
        prompt = f"""
        Competition: {slug}
        Search Context: {search_results}
        
        Task: Perform a deep-dive data architectural analysis.
        - Structure: Breakdown training/testing sets, sample submission, and supplementary files.
        - Target: Deep analysis of the target variable and its distribution.
        - Features: Key numerical/categorical/text/image features to watch for.
        - Evaluation: Critical analysis of the metric (e.g., LogLoss, MAE, F1) and what it implies for model bias.
        Format with professional data-science headers.
        """
        data_desc = call_llm(prompt)
        
        return {
            "data": data_desc
        }

    # Node 3: Elite Approaches (The Top 3)
    def approaches_node(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} similar competitions winning solutions historical SOTA")
        
        prompt = f"""
        Competition: {slug}
        Search Context: {search_results}
        
        Task: Architect the TOP 3 FINEST approaches for this competition.
        Analyze similar historical Kaggle competitions (e.g., 'past versions of {slug}' or 'similar domain competitions').
        
        For each of the 3 approaches, provide:
        - 🔥 Name: A catchy, professional name for the strategy.
        - 🛠️ Architecture: The model stack (e.g., XGBoost + CNN + Transformer).
        - 💡 Logic: Why this is a winning strategy for THIS data.
        
        🎯 CONCLUDE WITH: The "Most Feasible Grandmaster Approach" for an immediate start.
        """
        approaches = call_llm(prompt)
        
        return {
            "approaches": approaches
        }

    # Node 4: Grandmaster Secret Sauce
    def winners_node(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} winner solution writeup secret sauce tricks validation leak")
        
        prompt = f"""
        Competition: {slug}
        Search Context: {search_results}
        
        Task: Extract the "Secret Sauce" from previous winners or top solutions in this domain.
        Focus on the "1% differences" - the tiny tricks that push a model to 1st place:
        - Validation strategies that prevent overfitting.
        - Specific feature engineering (e.g., target encoding, time-series lags, image augmentations).
        - Post-processing or ensemble blending techniques.
        """
        winners_info = call_llm(prompt)
        
        return {
            "winners": winners_info
        }

    # Node 5: Universal Community Intel
    def discussion_node(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"Kaggle {slug} forum discussion bugs tips notebooks gold")
        
        prompt = f"""
        Competition: {slug}
        Search Context: {search_results}
        
        Task: Sift through the Kaggle community buzz for {slug}.
        - Critical Bugs: Are there data leaks, metric issues, or hidden files being discussed?
        - Community Observations: What are the current 'hot takes' on the leaderboard?
        - Golden Tips: Actionable advice shared by Kaggle Grandmasters in the forums.
        Make this feel like a high-level intelligence briefing.
        """
        discussion_info = call_llm(prompt)
        
        return {
            "discussion": discussion_info
        }


    workflow.add_node("explanation", explanation_node)
    workflow.add_node("data", data_node)
    workflow.add_node("approaches", approaches_node)
    workflow.add_node("winners", winners_node)
    workflow.add_node("discussion", discussion_node)

    workflow.set_entry_point("explanation")
    workflow.add_edge("explanation", "data")
    workflow.add_edge("data", "approaches")
    workflow.add_edge("approaches", "winners")
    workflow.add_edge("winners", "discussion")
    workflow.add_edge("discussion", END)

    return workflow.compile()
