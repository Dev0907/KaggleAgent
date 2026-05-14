# KaggleAgent.AI
**The Grandmaster's Intelligence Hub**  
*Built & Engineered by devparikh*

> "A small thing done well beats a big thing done badly."

KaggleAgent.AI is an autonomous, multi-agent intelligence platform designed to solve an obvious but painful problem: **Kaggle Information Overload.** For every competition, a data scientist must wade through hundreds of forum posts, dozens of code notebooks, and complex data documentation. 

This agent mimics the exact sequential thought process of a Kaggle Grandmaster to distill that noise into actionable competitive intelligence in under 3 minutes.

---

## 🎯 Why This? (Taste & Judgment)
In building KaggleAgent.AI, the focus was on **Distillation over Decoration**.
- **What’s Included**: A strictly orchestrated 6-agent pipeline using **LangGraph**. Each stage (Overview → Data → Approaches → Winners → Forum → Code Scout) is dependent on the previous, ensuring the AI "reasons" through the competition rather than just summarizing it.
- **What’s Left Out**: We avoided generic chat interfaces as the primary entry point. Instead, we built a **deterministic execution graph** that guarantees a complete analysis every time.
- **Smallest Interesting Version**: The core "V1" focus was the streaming graph UI and the context-aware "Ask the Grandmaster" chat—providing a functional end-to-end mentor that works universally on any Kaggle URL.

## 🛠️ Originality & Architecture
Unlike basic LLM wrappers, KaggleAgent.AI uses a **multi-agent state machine**:
1. **Overview Agent**: Identifies technical "System Problems" (noise, distribution shifts).
2. **Data Agent**: Audits dataset architecture and evaluation metrics.
3. **Winners Agent**: Extracts the "Secret Sauce" from the Top 5 historical solutions.
4. **Code Scout**: Autonomously hunts for high-fidelity GitHub repos and research papers.
5. **Grandmaster Chat**: A context-aware expert that "lives" in the analysis, ready for deep-dive Python questions.

## 🚀 Shipping Ability (It Works)
The project is built to be used **now**. 
- **Real-time Streaming**: Uses Server-Sent Events (SSE) to stream analysis results as they happen.
- **Dynamic UI**: A reactive dashboard with a visual execution pipeline.
- **Export Ready**: One-click professional Markdown report generation for offline sharing.

---

## 🏃 Quick Start (short & brief)

### 1. Requirements
Ensure you have Python 3.10+ and a `.env` in the `/backend` folder:
```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 2. Launch Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Launch Frontend
Open `frontend/index.html` in any modern browser.

---
*Built for the AI Build Challenge. Original work by devparikh.*
