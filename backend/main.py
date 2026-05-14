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
    model: str

graph = create_kaggle_agent()

async def event_generator(request: RunRequest):
    initial_state = {
        "competition": request.url,
        "model": request.model,
        "messages": [HumanMessage(content=f"Analyze competition: {request.url}")],
        "slug": "",
        "overview_summary": "",
        "data_description": "",
        "key_discussion_points": [],
        "past_winners_strategies": [],
        "final_strategy": ""
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
                if "overview_summary" in state_update and state_update["overview_summary"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "overview",
                        "content": state_update["overview_summary"]
                    }) + "\n"
                
                if "data_description" in state_update and state_update["data_description"]:
                    yield json.dumps({
                        "type": "content_update",
                        "section": "data",
                        "content": state_update["data_description"]
                    }) + "\n"
                
                if "key_discussion_points" in state_update and state_update["key_discussion_points"]:
                    points_html = "<ul>" + "".join([f"<li>{p}</li>" for p in state_update["key_discussion_points"]]) + "</ul>"
                    yield json.dumps({
                        "type": "content_update",
                        "section": "discussions",
                        "content": points_html
                    }) + "\n"
                
                if "past_winners_strategies" in state_update and state_update["past_winners_strategies"]:
                    strategy_content = "<br><br>".join(state_update["past_winners_strategies"])
                    yield json.dumps({
                        "type": "content_update",
                        "section": "strategies",
                        "content": strategy_content
                    }) + "\n"
                
                if "messages" in state_update and state_update["messages"]:
                    last_msg = state_update["messages"][-1].content
                    yield json.dumps({
                        "type": "log",
                        "message": f"[{node_name}] {last_msg}"
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
