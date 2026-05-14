import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from .agent import create_kaggle_agent
except ImportError:
    from agent import create_kaggle_agent

from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Kaggle Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you can replace this with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Kaggle Agent API is running"}

class RunRequest(BaseModel):
    url: str

graph = create_kaggle_agent()

async def event_generator(request: RunRequest):
    initial_state = {
        "competition": request.url,
        "messages": [HumanMessage(content=f"Analyze competition: {request.url}")],
        "slug": "",
        "explanation": "",
        "data": "",
        "approaches": "",
        "winners": "",
        "discussion": "",
        "external_links": ""
    }
    
    yield json.dumps({"type": "log", "message": f"Starting task for URL: {request.url}"}) + "\n"
    
    try:
        for output in graph.stream(initial_state):
            for node_name, state_update in output.items():
                
                # Yield state update for the frontend node graph
                yield json.dumps({
                    "type": "state_update",
                    "node": node_name
                }) + "\n"
                
                # Send the actual data components to display in the UI cards
                sections = ["explanation", "data", "approaches", "winners", "discussion", "external_links"]
                for section in sections:
                    if section in state_update and state_update[section]:
                        yield json.dumps({
                            "type": "content_update",
                            "section": section,
                            "content": state_update[section]
                        }) + "\n"
                
                await asyncio.sleep(0.5)
                
        yield json.dumps({
            "type": "result",
            "output": "Kaggle Agent completed all pipeline stages successfully."
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"

class ChatRequest(BaseModel):
    message: str
    context: str

@app.post("/api/chat")
async def chat_with_grandmaster(request: ChatRequest):
    try:
        from .agent import call_llm
    except ImportError:
        from agent import call_llm
    prompt = f"""
    You are a Kaggle Grandmaster and Python Expert.
    User Question: {request.message}
    
    CONTEXT ABOUT THE COMPETITION:
    {request.context}
    
    TASK: Answer the user's question with deep technical expertise. 
    - If they ask about python code, provide clean, optimized snippets.
    - If they ask about strategies, refer to the provided context and historical winning patterns.
    - Be professional, detailed, and acts as a high-level mentor.
    """
    response = call_llm(prompt)
    return {"response": response}

@app.post("/api/run")
async def run_agent(request: RunRequest):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
