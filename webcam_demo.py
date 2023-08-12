import argparse

import os
from sys import platform
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from model import Predictor
from omegaconf import OmegaConf
from einops import rearrange
import json

class PredictionService():

    def __init__(self):
        args = self._parse_args()
        self._model = self.init_model(args.config_path)

    def init_model(self, config_path):
        """
        Initialize the model using the provided configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Predictor: Initialized instance of the Predictor class.
        """
        try:
            with open(config_path, "r") as read_content:
                config = json.load(read_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at path: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding the configuration file: {config_path}")

        try:
            cfg = OmegaConf.create(
                {
                    "path_to_model": config["model"],
                    "path_to_class_list": config["class_list"],
                    "threshold": config["threshold"],
                    "topk": config["topk"],
                }
            )
            model = Predictor(cfg)
            return model
        except KeyError as e:
            raise KeyError(f"Missing key in configuration file: {e}")
        except ValueError as e:
            raise ValueError(f"Error creating Predictor configuration: {e}")



    def _parse_args(self):
        parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
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
        return args


    async def get_frame_results(self, frame_queue):
        if len(frame_queue) == 0:
            return
        cur_windows = list(np.array(frame_queue))
        results = self._model.predict(cur_windows)
        if not results:
            return None
        return results["labels"]




