from abc import ABC
from typing import List, Tuple

import jax.numpy as jnp 
import numpy as np
from omegaconf import DictConfig


class prediction_base(ABC):
    def __init__(self, cfg, model_type):
        self.cfg = cfg
        self.model_type = model_type

    def load_model(self, path: str, model_object) -> None:
        pass

    def load_prediction_methods(self, prediction_string:str) -> None:
        pass

    def return_prediction_function(self, framework:str) -> None:
        pass

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

class predictor(prediction_base):
    def __init__(self, cfg, model_type):
        super().__init__(cfg, model_type)

    def load_model(self, path: str, model_object) -> None:
        pass

    def load_prediction_methods(self, prediction_string:str) -> None:
        pass

    def return_prediction_function(self, framework:str) -> None:
        pass

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        pass