import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import create_kaggle_agent
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
        "discussion": ""
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
                if "explanation" in state_update and state_update["explanation"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "explanation",
                        "content": state_update["explanation"]
                    }) + "\n"
                
                if "data" in state_update and state_update["data"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "data",
                        "content": state_update["data"]
                    }) + "\n"
                
                if "approaches" in state_update and state_update["approaches"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "approaches",
                        "content": state_update["approaches"]
                    }) + "\n"
                
                if "winners" in state_update and state_update["winners"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "winners",
                        "content": state_update["winners"]
                    }) + "\n"
                
                if "discussion" in state_update and state_update["discussion"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "discussion",
                        "content": state_update["discussion"]
                    }) + "\n"
                
                await asyncio.sleep(0.5)
                
        yield json.dumps({
            "type": "result",
            "output": "Kaggle Agent completed all pipeline stages successfully."
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"

@app.post("/api/run")
async def run_agent(request: RunRequest):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
