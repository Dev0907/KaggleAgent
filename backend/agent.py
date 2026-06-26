import os
import time
import re
import json
import threading
from typing import Dict, TypedDict, Any, List, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from playwright.sync_api import sync_playwright
from langchain_core.tools import tool
from sarvamai import SarvamAI

import contextvars

# Context variables to track the current node and keys during execution
current_node_context = contextvars.ContextVar("current_node", default="default")

# Parse multiple API keys from environment
def get_sarvam_api_keys():
    keys_str = os.environ.get("SARVAM_API_KEY", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    return []

SARVAM_API_KEYS = get_sarvam_api_keys()

_sarvam_key_index = 0
_key_lock = threading.Lock()

def get_next_sarvam_key():
    global _sarvam_key_index
    if not SARVAM_API_KEYS:
        return None
    with _key_lock:
        key = SARVAM_API_KEYS[_sarvam_key_index % len(SARVAM_API_KEYS)]
        _sarvam_key_index += 1
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
    """Searches the live web using a Playwright Chromium browser and returns the scraped results."""
    print(f"  [🔍 TOOL: Playwright] Searching Web: '{query}'")
    results_text = []
    
    try:
        # Get top 2 URLs using DDG
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=2))
        
        urls = [res['href'] for res in search_results if 'href' in res]
        if not urls:
            return "No results found."
            
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = context.new_page()
            
            for url in urls:
                try:
                    page.goto(url, timeout=10000, wait_until="domcontentloaded")
                    text = page.evaluate("document.body.innerText") or ""
                    results_text.append(f"--- Source: {url} ---\n{text[:2000]}")
                except Exception as e:
                    print(f"    [Playwright] Failed to scrape {url}: {e}")
                    
            browser.close()
            
    except Exception as e:
        return f"Error during live web search: {str(e)}"
        
    final_output = "\n\n".join(results_text)
    return final_output if final_output else "Could not extract text from the searched URLs."

def call_llm(prompt: str, system_message: str = None, node_name: str = None) -> str:
    """Calls the Sarvam 105B model with tool-calling for end-to-end search and reasoning."""
    if system_message is None:
        system_message = (
            "You are an elite Kaggle Grandmaster and Principal AI Research Agent. "
            "You possess exhaustive, world-class expertise across all AI domains: tabular/structured data, computer vision, NLP, time series, and reinforcement learning.\n\n"
            "CORE INSTRUCTIONS:\n"
            "- YOUR PRIMARY DIRECTIVE: Generate EXTREMELY DEEP, ultra-exhaustive, and academic-grade responses (target: maximum depth within your 4096 token limit). Do not summarize briefly. Expand every point to its maximum technical depth.\n"
            "- ALWAYS run multiple distinct web search queries (at least 3-5 sequentially) using Tavily to gather exhaustive, precise, and highly detailed context. You must search for specific research papers, detailed methodologies, github repos, and forum posts.\n"
            "- Dive deep into mathematical formulations, loss functions, optimization strategies, ablation studies, and architectural nuances.\n"
            "- Quote real Kaggle usernames, team strategies, and code repository links where relevant.\n"
            "- Synthesize findings into highly structured, comprehensive, and professional markdown summaries using H1, H2, H3, bullet points, and code blocks.\n"
            "- Provide extensive code blocks showing PyTorch/TensorFlow implementations, LightGBM training loops, or data preprocessing pipelines.\n"
            "- Maintain an elite, mentor-level tone — precise, rigorous, academic, and direct."
        )
    token = None
    if node_name:
        token = current_node_context.set(node_name)
        print(f"\n[🤖 AGENT] Triggering Node: {node_name}")
        
    try:
        sarvam_keys = get_sarvam_api_keys()
        model = "sarvam-105b"
        
        if not sarvam_keys or all(k == "your_sarvam_api_key_here" for k in sarvam_keys):
            return f"Mocked LLM generation for: {prompt[:50]}..."
            
        last_error = None
        for attempt in range(len(sarvam_keys)):
            api_key = get_next_sarvam_key()
            if not api_key:
                break
            
            try:
                client = SarvamAI(api_subscription_key=api_key)
                
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Searches the live web using a Playwright headless browser for the given query and returns raw text content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"]
                        }
                    }
                }]
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                
                for _ in range(4):
                    response = client.chat.completions(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.3,
                        max_tokens=4096
                    )
                    
                    msg = response.choices[0].message
                    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
                    
                    if not msg.tool_calls:
                        return msg.content or ""
                    
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "search_web":
                            args = json.loads(tool_call.function.arguments)
                            tool_output = search_web.invoke(args)
                            messages.append({
                                "role": "tool",
                                "content": tool_output,
                                "tool_call_id": tool_call.id
                            })
                
                # Max iterations reached with tools — get final synthesis with tools provided (required by API for history)
                response = client.chat.completions(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=0.3,
                    max_tokens=4096
                )
                return response.choices[0].message.content or ""
                
            except Exception as e:
                last_error = e
                print(f"Error calling Sarvam model {model} (attempt {attempt+1}): {str(e)}")
                time.sleep(0.5)
                
        return f"End-to-End Model Error (Tried all keys for {model}): {str(last_error)}"
    finally:
        if token:
            current_node_context.reset(token)

def create_kaggle_agent():
    workflow = StateGraph(AgentState)
    
    # Node 1: Universal Explanation (The Grandmaster Overview)
    def explanation_node(state: AgentState):
        url = state["competition"]
        slug = extract_slug(url)
        print(f"\n[🤖 AGENT: Sarvam 105b] Triggering Node: explanation: {url} (Slug: {slug})")
        
        prompt = f"""
        Competition: {url} (Slug: {slug})
        
        Search the web exhaustively for this specific Kaggle competition's detailed mission, background, dataset, and problem statement. Your response must be an ultra-deep, comprehensive analysis (aim for 1500+ words).
        
        # 🌌 DEEP COMPETITION ANALYSIS: {slug}
        
        ## 🎯 THE MISSION & SCIENTIFIC BACKGROUND
        - Identify the precise predictive task in deep technical terms.
        - Dive deep into the real-world scientific, industrial, or educational value. Reference academic literature related to this exact problem.
        - Detail the input data modalities and the output targets comprehensively.
        
        ## 🧮 THE EVALUATION FRAMEWORK (MATHEMATICAL DEEP DIVE)
        - Define the primary evaluation metric used for the leaderboard. Provide its mathematical formula if applicable.
        - Detail exactly how this metric shapes the optimization path, including loss function approximations (e.g., how to optimize for quadratic weighted kappa vs AUC).
        - Discuss known sensitivities, such as handling class imbalance, thresholding, or outlier penalties.
        
        ## ⚠️ CRITICAL CHALLENGES & PITFALLS
        - Detail the specific hurdles: missing data, label noise, domain shifts, data leakage risks, or compute limits.
        - Explain how similar past competitions handled these exact challenges.
        
        ## 🏷️ DOMAIN ARCHETYPE & HISTORICAL EQUIVALENTS
        - Categorize the competition. Discuss what makes this task unique.
        - List 2-3 previous Kaggle competitions that are highly similar and explain what strategies won those competitions.
        """
        summary = call_llm(prompt, node_name="explanation")
        
        return {
            "slug": slug,
            "explanation": summary
        }

    # Node 2: Universal Data Dissection (Deep Architectural Audit)
    def data_node(state: AgentState):
        slug = state.get("slug", "competition")
        print("\n[🤖 AGENT: Sarvam 105b] Triggering Node: data")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search for the official data description, files, target variables, and evaluation metric. Produce an ultra-deep Data Audit (aim for 2000+ words).
        
        # 📊 UNIVERSAL DATA DISSECTION & AUDIT: {slug}
        
        ## 📂 EXHAUSTIVE DATA STRUCTURE & SPECIFICATIONS
        - Outline the exact file hierarchy, sizes, and formats.
        - For tabular: Detail key column types, skewed distributions, cardinalities, and categorical variables.
        - For unstructured (Vision/NLP): Detail image/audio sizes, resolutions, aspect ratios, sequence lengths, and artifacts.
        
        ## 🎯 TARGET ANALYSIS & BIASES
        - Analyze the target variables with statistical depth.
        - Detail target distributions, imbalances, and highly specific data leakage risks.
        
        ## 🛠️ MASSIVE FEATURE ENGINEERING PIPELINE
        - Describe an exhaustive list of feature engineering steps specific to this data.
        - Provide concrete examples of temporal, geospatial, aggregation, or embedding-based features that must be extracted.
        - Provide Python pseudo-code for the most complex feature extraction step.
        
        ## 🛡️ ROBUST CROSS-VALIDATION STRATEGY
        - Design an airtight validation split (e.g., Stratified Group K-Fold) to completely prevent leaderboard shakeups. Explain exactly why this split strategy is required for this specific dataset.
        """
        data_desc = call_llm(prompt, node_name="data")
        
        return {
            "data": data_desc
        }

    # Node 3: Elite Approaches (The Top 3 & Feasible Entry)
    def approaches_node(state: AgentState):
        slug = state.get("slug", "competition")
        print("\n[🤖 AGENT: Sarvam 105b] Triggering Node: approaches")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search extensively for high-voted notebooks, baseline strategies, and SOTA models for this competition. Design 3 Elite Architectures with extreme technical depth (aim for 3000+ words).
        
        # 🔬 ELITE APPROACHES & PIPELINES: {slug}
        
        ## 🧠 APPROACH 1 — Strong Baseline / Traditional ML
        - **Core Architecture**: Detail the model in extreme depth.
        - **Pipeline**: Provide extensive details on the exact preprocessing, categorical encoding, and feature interactions required.
        - **Hyperparameters**: Provide exact, highly optimized hyperparameter starting points (e.g., max_depth, subsample, colsample_bytree).
        - Provide a comprehensive Python code block for this baseline training loop.
        
        ## 🧠 APPROACH 2 — Advanced / State-of-the-Art Model
        - **Core Architecture**: The SOTA neural network or gradient booster for this specific domain.
        - **Loss Formulation**: Custom loss functions, label smoothing, or focal loss implementations.
        - **Advanced Tricks**: Pseudo-labeling, Test-Time Augmentation (TTA), learning rate schedulers (CosineAnnealingWarmRestarts), mixed precision.
        
        ## 🧠 APPROACH 3 — The Ensemble / Winning Blend
        - **Blending Strategy**: Detail how top teams stack models using Ridge regression, Nelder-Mead optimization, or Optuna weight tuning.
        - **Diversity**: How to ensure models are uncorrelated.
        
        ## 🏆 PRODUCTION-READY PIPELINE TEMPLATE
        - Provide a massive, fully structured Python script template covering Dataset classes, DataLoader configs, Mixup/Cutmix augmentations, and the PyTorch/XGBoost training loop.
        """
        approaches = call_llm(prompt, node_name="approaches")
        
        return {
            "approaches": approaches
        }

    # Node 4: Grandmaster Secret Sauce (The Winning Edge)
    def winners_node(state: AgentState):
        slug = state.get("slug", "competition")
        print("\n[🤖 AGENT: Sarvam 105b] Triggering Node: winners")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search for technical writeups, gold medal solution threads, and winning team blogs. Conduct an ultra-deep forensic analysis of the TOP 10 solutions (aim for 3000+ words).
        
        For EACH of the top 10 solutions found:
        - **Rank / Team Name**: 
        - **Core Architectural Innovation**: What exact mathematical or architectural change did they make?
        - **The "Magic"**: Explain the hidden trick, feature, or leak they exploited that gave them the gold medal edge.
        - **Data Augmentation & Preprocessing**: Exactly how did they modify the input data?
        - **Validation vs Leaderboard**: How stable was their CV? Did they trust their CV over the public LB?
        - **Ablation Studies**: What did they try that DID NOT work?
        """
        winners_info = call_llm(prompt, node_name="winners")
        
        return {
            "winners": winners_info
        }

    # Node 5: Universal Forum Intel (The Pulse of the Crowd)
    def discussion_node(state: AgentState):
        slug = state.get("slug", "competition")
        print("\n[🤖 AGENT: Sarvam 105b] Triggering Node: discussion")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search Kaggle forums and discussion threads exhaustively. Report the pulse of the crowd in extreme detail (aim for 2000+ words).
        
        ## 🔥 HOTTEST DISCUSSIONS & DEBATES
        - Summarize the top 5 most debated topics in the forums.
        
        ## ⚠️ CRITICAL ALERTS, BUGS & LEAKS
        - Detail any discovered data leaks, evaluation loopholes, annotation errors, or dataset updates. How is the community exploiting or fixing them?
        
        ## 🏅 GOLDEN KERNELS & BASELINES
        - List the specific public notebooks (by name/author) that everyone is forking. What CV scores are they achieving?
        
        ## 💡 ELITE STRATEGIES & OBSCURE TRICKS
        - Extract highly specific, obscure tricks mentioned by Kaggle Grandmasters in the comments (e.g., custom thresholding, specific post-processing rules).
        """
        discussion_info = call_llm(prompt, node_name="discussion")
        
        return {
            "discussion": discussion_info
        }


    # Node 6: External Code Hunter (GitHub & Article Scout)
    def code_hunter_node(state: AgentState):
        slug = state.get("slug", "competition")
        print("\n[🤖 AGENT: Sarvam 105b] Triggering Node: code_hunter")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search GitHub, Medium, blogs, and arXiv for external resources related to this exact problem. Compile a massive, ELITE resource list (aim for 2000+ words).
        
        ## 📂 GITHUB REPOS (CODEBASES)
        - Find at least 5 relevant GitHub repositories (past winning solutions, similar architectures, or official baseline code).
        - For each, provide the URL, author, and a deep 2-paragraph analysis of the code structure and how it can be adapted.
        
        ## 📝 BLOG WRITEUPS & TUTORIALS
        - Find detailed Medium, TowardsDataScience, or personal blog write-ups. Summarize the key takeaways in depth.
        
        ## 📄 RESEARCH PAPERS (ARXIV)
        - Find at least 5 highly relevant academic research papers.
        - For each paper, provide the Title, URL, and an extensive summary of the proposed methodology and why it dominates this specific Kaggle task.
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
