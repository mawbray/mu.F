from abc import ABC
from typing import List, Tuple

import jax.numpy as jnp 
import numpy as np
from omegaconf import DictConfig


class prediction_base(ABC):
    def __init__(self, cfg, model_type):
        self.cfg = cfg
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]

    def return_prediction_function(self, framework:str) -> None:
        pass

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

class predictor(prediction_base):
    def __init__(self, cfg, model_type):
        super().__init__(cfg, model_type)

    def load_trained_model(self, trainer):
        self.trainer = trainer

    def return_standardisation_metrics(self, string:str) -> None:

        assert string in ['input', 'output'], "string must be either 'input' or 'output' to return the standardisation metrics for the input or output data."
        
        if string == 'input':
            return self.trainer.standardisation_metrics_input
        elif string == 'output':
            return self.trainer.standardisation_metrics_output

    def return_prediction_function(self, string:str) -> None:
        
        assert string in ['standardised_model', 'unstandardised_model'], "string must be either 'standardised_model' or 'unstandardised_model' to return the prediction function for the standardised or unstandardised model."

        if string == 'standardised_model':
            return self.trainer.standardised_model
        elif string == 'unstandardised_model':
            return self.trainer.unstandardised_model
        

    def predict(self, prediction_function, X: jnp.ndarray) -> jnp.ndarray:
        return prediction_function(X)
    
    def get_serialised_model_data(self) -> Tuple:
        return self.trainer.get_serialised_model_data()