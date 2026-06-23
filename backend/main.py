import os
import sys
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

try:
    from .agent import create_kaggle_agent, call_llm
except ImportError:
    from agent import create_kaggle_agent, call_llm

from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Kaggle Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
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
        loop = asyncio.get_running_loop()
        
        # Create an iterator from the synchronous graph stream
        stream_iterator = graph.stream(initial_state)
        
        while True:
            try:
                # Retrieve the next output in a non-blocking thread executor
                output = await loop.run_in_executor(None, next, stream_iterator)
            except StopIteration:
                break
                
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
                
                await asyncio.sleep(0.1)
                
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
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: call_llm(prompt, node_name="chat"))
    return {"response": response}

@app.post("/api/run")
async def run_agent(request: RunRequest):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
