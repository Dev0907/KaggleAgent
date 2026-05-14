# KaggleAgent.AI - Elite Universal Analyzer
Developed & Engineered by **devparikh**

KaggleAgent.AI is a production-grade, multi-agent AI framework designed to dissect Kaggle competitions and provide actionable, Grandmaster-level insights. It utilizes a sophisticated pipeline to analyze competition objectives, data structures, historical winning strategies, and real-time community sentiment.

## 🚀 Key Features
- **Grandmaster Intelligence**: Powered by Llama 3.3 (70B) via Groq for deep technical reasoning.
- **Real-Time Research**: Integrated with Tavily for autonomous web searching of the latest Kaggle forums and technical writeups.
- **5-Agent Pipeline**:
  1. **Overview Agent**: Dissects the mission and identifies critical "system problems."
  2. **Data Agent**: Performs a deep architectural audit of datasets and evaluation metrics.
  3. **Strategy Agent**: Architects the most feasible top-tier approaches.
  4. **Winners Agent**: Extracts "Secret Sauce" from the Top 5 historical winning solutions.
  5. **Forum Agent**: Monitors hot topics and upvoted community intel.
- **Premium UI**: A sleek, modern dashboard with "Outfit" typography and real-time execution tracking.

## 🛠️ Technology Stack
- **Backend**: Python, FastAPI, LangGraph, LangChain, Groq API, Tavily API.
- **Frontend**: Vanilla HTML5, CSS3 (Modern Flex/Grid), Javascript (ES6+).
- **Environment**: Virtualized via `kegel` (Python venv).

## 🏃 Quick Start

### 1. Configure Environment
Create a `.env` file in the `backend/` directory:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 2. Install & Run
```bash
# Navigate to backend
cd backend

# Install dependencies (ensure you are in your venv)
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

### 3. Launch Frontend
Simply open `frontend/index.html` in your browser.

---
*Built with ❤️ for the Kaggle Community by devparikh.*
