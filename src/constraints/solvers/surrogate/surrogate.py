from abc import ABC
from typing import Tuple

from omegaconf import DictConfig
import jax.numpy as jnp

from constraints.solvers.surrogate.predictor import predictor
from constraints.solvers.surrogate.trainer import trainer, rebuilder


class surrogate_base(ABC):
    def __init__(self, graph, unit_index: int, cfg: DictConfig, model_type: str, iterate: int) -> None:

        self.cfg = cfg
        self.graph = graph
        self.unit_index = unit_index
        assert type(model_type) == tuple, "model_type must be a tuple of strings"
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]
        self.iterate = iterate

        assert self.model_class in ["regression", "classification"], "model_class must be either 'regression' or 'classification'"
        if self.model_class == "regression":
            assert self.model_subclass in ["ANN", "GP"], "regression model_subclass must be either 'ANN' or 'GP'"
        elif self.model_class == "classification":
            assert self.model_subclass in ["ANN", "SVM"], "classifier model_subclass must be either 'ANN', or 'SVM'"

        assert self.model_surrogate in ["live_set_surrogate", "probability_map_surrogate", "forward_evaluation_surrogate"], "model_surrogate must be one of ['live_set_surrogate', 'probability_map_surrogate', 'forward_evaluation_surrogate'] indicating a parameterisation of the feasible region, probability map or unit dynamics respectively."

    def fit(self) -> None:
        pass

    def predict(self, string: str, X: jnp.ndarray) -> jnp.ndarray:
        pass

    def get_model(self, string:str) -> callable:
        pass

        

class surrogate(surrogate_base):
    def __init__(self, graph, unit_index, cfg: DictConfig, model_type: str, iterate:int) -> None:
        super().__init__(graph, unit_index, cfg, model_type, iterate)
        self.model = None
        self.trainer = trainer(graph, unit_index, cfg, model_type, iterate)
        self.predictor = predictor(cfg, model_type)

    def fit(self, node=None) -> None:
        self.trainer.train(node)
        self.predictor.load_trained_model(self.trainer)

    def predict(self, string:str, X: jnp.ndarray) -> jnp.ndarray:
        return self.predictor.predict(self.get_model(string), X)

    def get_model(self, string: str) -> callable:
        return self.predictor.return_prediction_function(string)
    
    def get_serailised_model_data(self) -> Tuple:
        return self.predictor.get_serialised_model_data()
    

class surrogate_reconstruction(ABC):
    def __init__(self, cfg: DictConfig, model_type: str, problem_data: dict) -> None:
        self.cfg = cfg
        self.model_type = model_type
        self.problem_data = problem_data

        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]

        assert self.model_class in ["regression", "classification"], "model_class must be either 'regression' or 'classification'"
        if self.model_class == "regression":
            assert self.model_subclass in ["ANN", "GP"], "regression model_subclass must be either 'ANN' or 'GP'"
        elif self.model_class == "classification":
            assert self.model_subclass in ["ANN", "SVM"], "classifier model_subclass must be either 'ANN', or 'SVM'"

        assert self.model_surrogate in ["live_set_surrogate", "probability_map_surrogate", "forward_evaluation_surrogate"], "model_surrogate must be one of ['live_set_surrogate', 'probability_map_surrogate', 'forward_evaluation_surrogate'] indicating a parameterisation of the feasible region, probability map or unit dynamics respectively."


    def rebuild_model(self):
        return rebuilder(self.cfg, self.model_type, self.problem_data).rebuild()
