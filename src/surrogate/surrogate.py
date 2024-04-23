from abc import ABC
from typing import Tuple

from omegaconf import DictConfig
import jax.numpy as jnp

from predictor import predictor
from trainer import trainer


class surrogate_base(ABC):
    def __init__(self, cfg: DictConfig, model_type: str) -> None:

        self.cfg = cfg
        assert type(model_type) == tuple, "model_type must be a tuple of strings"
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]

        assert self.model_class in ["regression", "classification"], "model_class must be either 'regression' or 'classification'"
        if self.model_class == "regression":
            assert self.model_subclass in ["ANN", "GP"], "regression model_subclass must be either 'ANN' or 'GP'"
        elif self.model_class == "classification":
            assert self.model_subclass in ["ANN", "SVM"], "classifier model_subclass must be either 'ANN', or 'SVM'"

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        pass

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

    def update(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def get_data(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass

    def get_model(self):
        pass

    def get_model_type(self) -> str:
        return self.model_type
        

class surrogate(surrogate_base):
    def __init__(self, cfg: DictConfig, model_type: str) -> None:
        super().__init__(cfg, model_type)
        self.model = None
        self.trainer = None
        self.predictor = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        pass

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

    def update(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def get_data(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass

    def get_model(self):
        pass