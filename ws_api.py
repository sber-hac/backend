from fastapi import FastAPI, WebSocket
import requests
import collections
import asyncio
import uvicorn
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI()
queue = collections.deque()

@app.websocket("/ws/result")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if len(queue) != 0:
            await websocket.send_text(queue.pop())
        await asyncio.sleep(1)

class Item(BaseModel):
    result:str

@app.post("/sendRecognizedMessage")
async def send_recognized_message(request : Item):
    queue.append(request.result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app=app, port=8081)