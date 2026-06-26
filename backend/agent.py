import os
import time
import re
import json
import threading
from typing import Dict, TypedDict, Any, List, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
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

def get_tavily_api_keys():
    keys_str = os.environ.get("TAVILY_API_KEY", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    return []

SARVAM_API_KEYS = get_sarvam_api_keys()
TAVILY_API_KEYS = get_tavily_api_keys()

_sarvam_key_index = 0
_tavily_key_index = 0
_key_lock = threading.Lock()
_tavily_lock = threading.Lock()

def get_next_sarvam_key():
    global _sarvam_key_index
    if not SARVAM_API_KEYS:
        return None
    with _key_lock:
        key = SARVAM_API_KEYS[_sarvam_key_index % len(SARVAM_API_KEYS)]
        _sarvam_key_index += 1
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
    tavily_keys = get_tavily_api_keys()
    
    if not tavily_keys or all(k == "your_tavily_api_key_here" for k in tavily_keys):
        env_key = os.environ.get("TAVILY_API_KEY")
        if env_key:
            tavily_keys = [env_key]
            
    if not tavily_keys or all(k == "your_tavily_api_key_here" for k in tavily_keys):
        return f"Mocked search results for: {query}"
        
    last_error = None
    # Try up to the number of keys we have available
    for _ in range(max(1, len(tavily_keys))):
        api_key = get_next_tavily_key()
        if not api_key:
            break
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, search_depth="advanced")
            results = "\n".join([res['content'] for res in response.get('results', [])])
            if results:
                return results
        except Exception as e:
            last_error = e
            print(f"Tavily search error with key {api_key[:10]}...: {str(e)}")
            
    return f"Search error (tried available keys): {str(last_error)}"

def call_llm(prompt: str, system_message: str = None, node_name: str = None) -> str:
    """Calls the Sarvam 105B model with tool-calling for end-to-end search and reasoning."""
    if system_message is None:
        system_message = (
            "You are an elite Kaggle Grandmaster and Principal AI Research Agent. "
            "You possess deep expertise across all AI domains: tabular/structured data, computer vision, NLP, time series, and reinforcement learning.\n\n"
            "CORE INSTRUCTIONS:\n"
            "- ALWAYS run multiple distinct web search queries (at least 2-3 sequentially) to gather exhaustive, precise context. "
            "First query the specific competition target; then query notebooks, discussions, and external Kaggle solutions.\n"
            "- For tabular data (like Titanic): Focus on feature engineering, missing value imputation, random forests, XGBoost/LightGBM, and robust cross-validation.\n"
            "- For deep learning (CV/NLP): Cite exact architectures, loss functions, custom metrics, and optimization tricks.\n"
            "- Quote real Kaggle usernames, team strategies, and code repository links where relevant.\n"
            "- Synthesize findings into highly structured, comprehensive, and professional markdown summaries.\n"
            "- Maintain an elite, mentor-level tone — precise, rigorous, and direct."
        )
    token = None
    if node_name:
        token = current_node_context.set(node_name)
        
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
                        "description": "Searches the web for the given query using Tavily and returns the results as a string.",
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
                        max_tokens=4000
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
                    max_tokens=4000
                )
                return response.choices[0].message.content or ""
                
            except Exception as e:
                last_error = e
                print(f"Error calling Sarvam model {model} with key index {key_idx}: {str(e)}")
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
        
        prompt = f"""
        Competition: {url} (Slug: {slug})
        
        Search the web for this specific Kaggle competition's detailed mission, background, dataset, and problem statement. Synthesize into:
        
        # 🌌 DEEP COMPETITION ANALYSIS: {slug}
        
        ## 🎯 THE MISSION
        - Identify the precise predictive task (e.g., binary classification, object detection, time series forecasting).
        - Detail the input data modalities (e.g., Tabular CSVs, Images, Audio, Text) and the output targets.
        - Discuss the real-world scientific, industrial, or educational value.
        
        ## 🧮 THE EVALUATION FRAMEWORK
        - Define the primary evaluation metric used for the leaderboard.
        - Detail how this metric shapes the optimization path and any known sensitivities (e.g., handling class imbalance, thresholding).
        
        ## ⚠️ CRITICAL CHALLENGES
        - Detail the specific hurdles: missing data, label noise, domain shifts, data leakage risks, or compute limits.
        
        ## 🏷️ DOMAIN ARCHETYPE
        - Categorize the competition (e.g., Tabular Beginner, Advanced NLP, Object Detection). Discuss what makes this task unique compared to standard datasets.
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
        Competition Slug: {slug}
        
        Search for the official data description, files, target variables, and evaluation metric. Produce a Deep Data Audit:
        
        # 📊 UNIVERSAL DATA DISSECTION & AUDIT: {slug}
        
        ## 📂 DATA STRUCTURE & SPECIFICATIONS
        - Outline the file hierarchy (directories, CSVs, metadata, image formats).
        - For tabular: Detail key column types, skewed distributions, and categorical variables.
        - For unstructured: Detail image/audio sizes, lengths, and formats.
        
        ## 🎯 TARGET ANALYSIS & BIASES
        - Analyze the target variables.
        - Detail target distributions, imbalances (e.g., survival rates for Titanic, rare disease classes), and potential data leakage risks.
        
        ## 🛠️ FEATURE ENGINEERING PRIORITIES
        - Describe essential feature engineering steps specific to this data (e.g., extracting titles from names in Titanic, creating family size, or standardizing image contrasts).
        
        ## 🛡️ ROBUST CROSS-VALIDATION STRATEGY
        - Design a robust validation split (e.g., Stratified K-Fold, GroupKFold by patient/origin, Time-Series Split) to mirror the public/private leaderboard split and prevent shakeups.
        """
        data_desc = call_llm(prompt, node_name="data")
        
        return {
            "data": data_desc
        }

    # Node 3: Elite Approaches (The Top 3 & Feasible Entry)
    def approaches_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search for high-voted notebooks, baseline strategies, and successful models specifically for this competition. Design 3 Elite Architectures or Pipelines:
        
        # 🔬 ELITE APPROACHES & PIPELINES: {slug}
        
        ## 🧠 APPROACH 1 — Strong Baseline / Traditional ML
        - **Core Architecture**: Simple but highly effective models (e.g., Random Forest, XGBoost, or Logistic Regression for tabular; ResNet34 for vision).
        - **Key Feature Interactions**: The most critical features or embeddings fed into the model.
        - **Hyperparameters**: Best initial starting points for learning rate, depth, or regularization.
        
        ## 🧠 APPROACH 2 — Advanced / State-of-the-Art Model
        - **Core Architecture**: The typical SOTA used for this specific domain (e.g., LightGBM/CatBoost ensembles, Transformers/BERT, or YOLO/UNet).
        - **Loss Formulation & Tuning**: Custom loss functions or optimization targets.
        - **Hyperparameters & Tricks**: Advanced tricks like pseudo-labeling, test-time augmentation (TTA), or specific learning rate schedules.
        
        ## 🧠 APPROACH 3 — The Ensemble / Winning Blend
        - **Core Architecture**: How top teams typically blend models for this archetype.
        - **Blending Strategy**: Weighted averaging, stacking with a meta-model, or voting classifiers.
        - **Diversity**: How to ensure the blended models make uncorrelated errors.
        
        ## 🏆 PRODUCTION-READY PIPELINE TEMPLATE
        - Provide step-by-step pseudo-code or Python template for the training loop and feature pipeline.
        """
        approaches = call_llm(prompt, node_name="approaches")
        
        return {
            "approaches": approaches
        }

    # Node 4: Grandmaster Secret Sauce (The Winning Edge)
    def winners_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search for technical writeups, gold medal solution threads, and winning team blogs. Profile the TOP 5 solutions:
        
        For each:
        - **Rank / Team**: 
        - **Core Idea**: (what made it work)
        - **Technical Stack**: (models, frameworks, hardware)
        - **CV vs LB**: (how stable was their validation?)
        - **Secret Sauce**: (the one thing that gave them the edge)
        """
        winners_info = call_llm(prompt, node_name="winners")
        
        return {
            "winners": winners_info
        }

    # Node 5: Universal Forum Intel (The Pulse of the Crowd)
    def discussion_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search Kaggle forums, discussion threads, and community posts. Report the pulse of the crowd:
        
        ## 🔥 HOTTEST DISCUSSIONS
        Top-voted threads and what they reveal.
        
        ## ⚠️ ALERTS & LEAKS
        Data leaks, evaluation loopholes, or competition updates.
        
        ## 🏅 GOLDEN KERNELS
        Community-endorsed notebooks/approaches that everyone references.
        
        ## 💡 ELITE STRATEGIES
        Strategies shared by top competitors in forum posts.
        """
        discussion_info = call_llm(prompt, node_name="discussion")
        
        return {
            "discussion": discussion_info
        }


    # Node 6: External Code Hunter (GitHub & Article Scout)
    def code_hunter_node(state: AgentState):
        slug = state.get("slug", "competition")
        
        prompt = f"""
        Competition Slug: {slug}
        
        Search GitHub, Medium, blogs, and arXiv for external resources. Compile an ELITE resource list:
        
        ## 📂 GITHUB REPOS
        Winning solution codebases with star counts and key features.
        
        ## 📝 BLOG WRITEUPS
        Technical deep-dives from winners or participants.
        
        ## 📄 RESEARCH PAPERS
        Papers that inspired winning approaches or are directly applicable.
        
        For each entry provide: title, URL, and a 1-sentence summary of why it matters.
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
