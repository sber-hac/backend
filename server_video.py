import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import webcam_demo
import numpy as np
import uvicorn
import threading
import aiohttp
import requests
import collections
import time

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

app = FastAPI()

frame_queue = []

service_prediction = webcam_demo.PredictionService("config.json")
full_queue_event = threading.Event()
recognize_result = ""

# @app.websocket("/result")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         if len(recognize_result_deq) != 0:
#             full_queue_event.clear()
#             await websocket.send_text(recognize_result_deq.pop())

def send_message(message):
    requests.post('http://localhost:8081/sendRecognizedMessage', json = {"result" : message})  

    

async def queue_is_full(frames):
    global recognize_result
    recognize_result = await service_prediction.get_frame_results(list(np.array(frames)))
    full_queue_event.set()
    print(recognize_result)
    if recognize_result:
        send_message(recognize_result[0])
    frame_queue.clear()


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        frame_after = VideoFrame.to_ndarray(frame, format="bgr24")
        img = np.array(cv2.resize(frame_after, (224, 224))[:,:,::-1])

        if (len(frame_queue) >= 32):
            await queue_is_full(frame_queue)
        frame_queue.append(img)

        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def websocket_result(request):
    print('Websocket connection starting')
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection ready')
    while full_queue_event.is_set():        
            full_queue_event.clear()
            if recognize_result:
                await ws.send_str(recognize_result)
    return ws



async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


    def _parse_args(self):
        parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
        
        args = parser.parse_args()
        return args
    
def start_uvicorn():
    global recognize_result
    uvicorn.run("server_video:app", host="127.0.0.1", port=5000, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")


    parser.add_argument('--config_path', default='config.json', help='model config')
    parser.add_argument(
            '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
            '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
            '--sample-length',
            type=int,
            default=32,
            help='len of frame queue')
    parser.add_argument(
            '--drawing-fps',
            type=int,
            default=20,
            help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
            '--inference-fps',
            type=int,
            default=4,
            help='Set upper bound FPS value of model inference')
    parser.add_argument(
            '--openvino',
            action='store_true',
            help='Use OpenVINO backend for inference. Available only on Linux')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    # app.router.add_route('GET', '/ws/result', websocket_result)
    # t1 = threading.Thread(target=start_uvicorn)
    # t1.start()
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
