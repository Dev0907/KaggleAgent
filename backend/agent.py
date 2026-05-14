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
    external_links: str

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
    
    # Node 1: Universal Explanation (The Grandmaster Overview)
    def explanation_node(state: AgentState):
        url = state["competition"]
        slug = extract_slug(url)
        
        # Searching for current objective and historical context
        search_results = search_web(f"Kaggle competition {slug} objective challenges system problems historical context")
        
        prompt = f"""
        Competition: {slug}
        URL: {url}
        Context: {search_results}
        
        TASK: Act as a Kaggle Grandmaster. Provide a Deep Strategic Analysis.
        1. THE MISSION: What is the specific predictive task?
        2. THE VALUE: Why does this matter? Real-world impact?
        3. CRITICAL SYSTEM PROBLEMS: Deep-dive into the "tricky" technical hurdles. Identify noise, data distribution shifts, non-stationarity, or memory constraints. Detail these "system problems" enough so a user understands exactly why this is a hard problem.
        4. DOMAIN ARCHETYPE: Is this a 'Universal' Time-Series, Computer Vision, or Tabular challenge?
        
        Format using professional markdown. Use callouts or bold text for emphasis.
        """
        summary = call_llm(prompt)
        
        return {
            "slug": slug,
            "explanation": summary
        }

    # Node 2: Universal Data Dissection (Deep Architectural Audit)
    def data_node(state: AgentState):
        slug = state.get("slug", "competition")
        # Targeting official data descriptions
        search_results = search_web(f"site:kaggle.com/competitions/{slug}/data Kaggle competition data description target variable")
        
        prompt = f"""
        Competition: {slug}
        Context: {search_results}
        
        TASK: Perform an Elite Data Audit.
        1. DATA STRUCTURE: Describe the file hierarchy (train/test/supplemental).
        2. TARGET ANALYSIS: What exactly are we predicting? Is it a continuous value (regression) or a class (classification)? 
        3. KEY FEATURES: Highlight critical feature groups (e.g., technical indicators, fundamental data, anonymized sensors).
        4. THE EVALUATION: Deep-dive into the metric (e.g., Sharpe Ratio, Pearson Correlation, RMSE). What does this metric reward (e.g., consistency vs. outlier accuracy)?
        5. DATA CHALLENGES: Identify leaks, missing values, or specific temporal constraints (e.g., no future-looking).
        """
        data_desc = call_llm(prompt)
        
        return {
            "data": data_desc
        }

    # Node 3: Elite Approaches (The Top 3 & Feasible Entry)
    def approaches_node(state: AgentState):
        slug = state.get("slug", "competition")
        # Targeting high-voted code and top solutions
        search_results = search_web(f"site:kaggle.com/competitions/{slug}/code?sort=votes Kaggle top solutions winning approaches strategy")
        
        prompt = f"""
        Competition: {slug}
        Context: {search_results}
        
        TASK: Architect 3 "Grandmaster" Approaches based on High-Voted Code & Solutions.
        Analyze similar historical competitions or the specific top-voted notebooks for {slug}.
        
        FOR EACH APPROACH PROVIDE:
        - 🔥 STRATEGY NAME: (e.g., "The Temporal Transformer Blend")
        - 🛠️ THE ARCHITECTURE: List the specific models (e.g., LightGBM, LSTM, TabNet).
        - 💡 THE RATIONALE: Why does this specific architecture beat the baseline?
        - 🧪 FEATURE ENGINEERING: One unique feature idea for this approach extracted from top-voted kernels.
        
        🎯 FINAL VERDICT: Define the "Universal Most Feasible Approach" for a fast-start submission that can hit the top 20%.
        """
        approaches = call_llm(prompt)
        
        return {
            "approaches": approaches
        }

    # Node 4: Grandmaster Secret Sauce (The Winning Edge)
    def winners_node(state: AgentState):
        slug = state.get("slug", "competition")
        # Specifically searching for top 5 solutions from this or previous versions
        search_results = search_web(f"Kaggle {slug} top 5 winning solutions previous years technical writeup gold medal")
        
        prompt = f"""
        Competition: {slug}
        Context: {search_results}
        
        TASK: Deep-Dive into the TOP 5 WINNING SOLUTIONS.
        If this competition was held before (or has similar archetypes), identify the Top 5 approaches that secured Gold.
        
        FOR EACH OF THE TOP 5 SOLUTIONS, DETAIL:
        1. 🏆 RANK & USER: Who was it?
        2. 🧠 CORE IDEA: The "Aha!" moment or unique architecture (e.g., GBDT-NN blend, Denoising Autoencoder).
        3. 🛠️ TECHNICAL STACK: Key libraries and models.
        4. 📈 CV vs LB: How did they ensure stable validation?
        5. ✨ THE "SECRET SAUCE": The specific trick (e.g., specific augmentation, post-processing) that gave them the edge.
        
        FORMAT: Use a clean, section-wise markdown layout with horizontal rules between solutions. Make it look professional and "Elite".
        """
        winners_info = call_llm(prompt)
        
        return {
            "winners": winners_info
        }

    # Node 5: Universal Forum Intel (The Pulse of the Crowd)
    def discussion_node(state: AgentState):
        slug = state.get("slug", "competition")
        # Targeting high-voted discussions and kernels specifically
        search_results = search_web(f"site:kaggle.com/competitions/{slug}/discussion?sort=votes Kaggle most upvoted forum topics strategy")
        
        prompt = f"""
        Competition: {slug}
        Context: {search_results}
        
        TASK: Provide a "Most Upvoted & Hot Community Intelligence Briefing".
        - THE TOP-VOTED BUZZ: What are the most upvoted discussions? Identify the specific "hot" topics Grandmasters are debating.
        - ELITE APPROACHES FROM FORUMS: Extract specific strategies or "meta-approaches" mentioned in highly upvoted comments.
        - CRITICAL ALERTS & LEAKS: Any high-voted alerts regarding data leakage, CV-LB discrepancy, or metric bugs?
        - GOLDEN KERNELS: Identify the most "starred" or "forked" notebooks and what unique logic they contain.
        - PROPER WORKFLOWS: Summarize the community-endorsed "proper way" to handle the validation for this specific competition.
        """
        discussion_info = call_llm(prompt)
        
        return {
            "discussion": discussion_info
        }


    # Node 6: External Code Hunter (GitHub & Article Scout)
    def code_hunter_node(state: AgentState):
        slug = state.get("slug", "competition")
        search_results = search_web(f"site:github.com Kaggle {slug} winning code github repo similar competitions")
        search_results += "\n" + search_web(f"Kaggle {slug} technical medium article writeup research paper")
        
        prompt = f"""
        Competition: {slug}
        Context: {search_results}
        
        TASK: Find and link to ELITE external resources.
        1. GITHUB REPOS: Find 3-5 high-quality GitHub repositories that implement winning or high-performing strategies for {slug} or very similar competitions.
        2. TECHNICAL ARTICLES: Find links to Medium, Substack, or personal blog writeups from Top 10 finishers.
        3. RESEARCH PAPERS: Are there specific papers (e.g., from ArXiv) that are being referenced as the backbone for this competition's best models?
        
        FORMAT: Use a clean list with [Title](URL) format. Provide a 1-sentence summary for WHY each link is valuable.
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
