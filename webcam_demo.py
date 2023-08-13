import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from model import Predictor
from omegaconf import OmegaConf
import json

class PredictionService():

    def __init__(self, config_path):
        self._model = self.init_model(config_path)

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



    async def get_frame_results(self, frame_queue):
        if len(frame_queue) == 0:
            return
        cur_windows = list(np.array(frame_queue))
        results = self._model.predict(cur_windows)
        if not results:
            return None
        return results["labels"]




