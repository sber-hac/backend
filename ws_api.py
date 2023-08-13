from fastapi import FastAPI, WebSocket
import collections
import asyncio
import uvicorn
from pydantic import BaseModel
import speechkit_service

app = FastAPI()
queue = collections.deque()
text_queue = collections.deque()

@app.websocket("/ws/result")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if len(queue) != 0:
            await websocket.send_text(queue.pop())
        await asyncio.sleep(1)

@app.websocket("/ws/recognizeAudioFile")
async def recognize_audio(websocket: WebSocket):
    await websocket.accept()
    while True:
        file = await websocket.receive_bytes()
        with open("record.wav", "wb") as f:
            f.write(file)
            result = speechkit_service.recognize("record.wav")
            await websocket.send_text(result)

@app.websocket("/ws/audioSpeechResult")
async def websocket_audio_result_endpoint(websocket : WebSocket):
    await websocket.accept()
    while True:
        if len(text_queue) != 0:
            await websocket.send_text(text_queue.pop())
        await asyncio.sleep(1)

class Item(BaseModel):
    result:str

@app.post("/sendRecognizedMessage")
async def send_recognized_message(request : Item):
    queue.append(request.result)

@app.post("/sendRecognizedAudioMessage")
async def send_recognized_message(request : Item):
    text_queue.append(request.result)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app=app, port=8081)