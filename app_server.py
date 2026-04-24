from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import os
from face_detect2 import VideoStream
from supervisor import supervisor_decision
from user_profile import identify_user, register_user, load_all_profiles

app = FastAPI(title="Casmocart AI")

# Singleton VideoStream
video_stream = VideoStream()

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    user_name: str
    query: str

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream.get_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/scan")
async def scan_face():
    # In web mode, we use the data from the live stream
    data = video_stream.latest_data
    if not data:
        return {"status": "error", "error": "No face detected in stream"}
    
    # Try to recognize
    signature = data.get('signature')
    name, profile = identify_user(signature)
    
    return {
        "status": "success",
        "name": name if name else "New User",
        "interpretation": data['interpretation'],
        "signature": signature,
        "raw_data": data
    }

@app.post("/analyze")
async def analyze(request: QueryRequest):
    profiles = load_all_profiles()
    profile = profiles.get(request.user_name)
    
    # Get the active scan data from the stream
    skin_data = video_stream.latest_data
    
    if profile:
        context_query = f"User: {request.user_name}, Concerns: {profile['user_info']['concerns']}. Query: {request.query}"
    else:
        context_query = request.query
        
    # CRITICAL: Pass skin_data to the supervisor so it can use the mesh analysis
    result = supervisor_decision(context_query, skin_data=skin_data)
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
