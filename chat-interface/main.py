from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from entities import invoke_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ WebSocket endpoint (THIS WAS MISSING)
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            user_message = await websocket.receive_text()
            print("User:", user_message)

            # Get LLM response
            bot_response = invoke_llm.generate_model_response(user_message)
            print("Bot:", bot_response)

            # Stream response character-by-character
            partial = ""
            for char in bot_response:
                partial += char
                await websocket.send_text(partial)
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected")