from abc import ABC
from typing import List, Tuple

import jax.numpy as jnp 
import numpy as np
from omegaconf import DictConfig
from gpjax import Dataset

from data_utils import binary_classifier_data_preparation, regression_data_preparation

class trainer_base(ABC):
    def __init__(self, graph, unit_index, cfg, model_type):
        self.cfg = cfg
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.graph = graph
        self.unit_index = unit_index


    def get_model(self, path: str, model_object) -> None:
        pass

    def load_trainer_methods(self, prediction_string:str) -> None:
        pass

    def train(self, dataset: jnp.ndarray) -> jnp.ndarray:
        pass

class trainer(trainer_base):
    def __init__(self, graph, unit_index, cfg, model_type):
        super().__init__(graph, unit_index, cfg, model_type)
        

    def get_model(self, model_object) -> None:
        pass

    def load_trainer_methods(self) -> None:
        if self.model_subclass == 'ANN':
            from nn_utils import train as train_ann
            self.trainer = train_ann
        elif self.model_subclass == 'GP':
            from gp_utils import train as train_gp
            self.trainer = train_gp
        elif self.model_subclass == 'SVM':
            from svm_utils import train as train_svm
            self.trainer = train_svm
        return 

    def get_data(self) -> None:
        if self.model_class == 'regression': # TODO make sure this method exists and acts on the right component of the graph e.g. edge or node
            dataset = regression_data_preparation(self.graph, self.unit_index, self.cfg)
        elif self.model_class == 'classification': # this is only used for determining node data
            data_points, labels = binary_classifier_data_preparation(self.graph, self.unit_index, self.cfg)
            if self.model_subclass == 'svm' : dataset = (data_points, labels)
            elif self.model_subclass == 'ANN' : dataset = Dataset(X=data_points, y=labels)
        return dataset
    


    def train(self) -> jnp.ndarray:
        pass