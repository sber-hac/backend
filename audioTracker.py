import wave

import av
import cv2
import numpy as np
import pyaudio as pydio
from av.audio.frame import AudioFrame
from pyaudio import PyAudio
import speechkit_service
import requests
from aiortc import MediaStreamTrack

class AudioTransformTrack(MediaStreamTrack):
    """
    A audio stream track that transforms frames from an another track.
    """

    kind = "audio"
    auth_token = 'MjVkMjE5MmMtNzBlMC00N2QwLTkyYmYtZDBjMmRhYTlhMDE4OjRkMWRmNGE3LTFjYjYtNGI4Yy05MzgwLWFiOTk0ODBmZWY1Ng=='

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.channels = 2
        self.sample_rate = 16000
        self.buffer_size = 1024
        self.p = PyAudio()
        self.frames = []


    async def recv(self):
        frame = await self.track.recv()
        frame_after = AudioFrame.to_ndarray(frame, format="s16", layout="mono")
        print("len", len(self.frames))
        if (len(self.frames) < 300):
            self.frames.append(frame_after)

        if (len(self.frames) >= 300):
            # added dummy code to save wave file to test able to grap mic data works fine
            self.frames.append(frame_after)
            wf = wave.open("recordon.wav", 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pydio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            print("recorded audio saved")
            try:
                #response = requests.post(url='https://smartspeech.sber.ru/rest/v1/speech:recognize',
                #          headers={'Authorization' : f'Bearer {self.auth_token}',
                #                    'Content-Type' : 'audio/x-pcm;bit=16;rate=16000'},
                #          files={'data-binary': open('record.wav', 'rb')}, verify=False)

                #print(response)
                self.frames.clear()
                result = speechkit_service.recognize("recordon.wav")
                print(result)
            except Exception as err:
                print(err)

        # print("audio phase 3")

        return frame

