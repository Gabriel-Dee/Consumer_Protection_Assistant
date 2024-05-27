from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from chatbot import get_chatbot_response
from langchain.memory import ChatMessageHistory
from uuid import uuid4

app = FastAPI()

# Allow CORS for all origins (for testing purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    session_id: str = None

# In-memory storage for chat histories
chat_histories: Dict[str, ChatMessageHistory] = {}

def get_or_create_session(session_id: str):
    if session_id is None or session_id not in chat_histories:
        session_id = str(uuid4())
        chat_histories[session_id] = ChatMessageHistory()
    return session_id, chat_histories[session_id]

@app.post("/chat")
async def chat(message: Message):
    try:
        session_id, chat_history = get_or_create_session(message.session_id)
        response, updated_chat_history = get_chatbot_response(message.text, chat_history)
        chat_histories[session_id] = updated_chat_history
        return {"response": response, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
