from abc import ABC
from typing import List, Tuple
from functools import partial

import jax.numpy as jnp 
import numpy as np
from omegaconf import DictConfig
from gpjax import Dataset

from surrogate.data_utils import binary_classifier_data_preparation, regression_node_data_preparation, forward_evaluation_data_preparation
from surrogate.gp_utils import train as train_gp
from surrogate.nn_utils import hyperparameter_selection as train_ann
from surrogate.svm_utils import train as train_svm


class trainer_base(ABC):
    def __init__(self, graph, unit_index, cfg, model_type, iterate):
        self.cfg = cfg
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]
        self.graph = graph
        self.unit_index = unit_index
        self.iterate = iterate


    def get_model(self, path: str, model_object) -> None:
        pass

    def load_trainer_methods(self, prediction_string:str) -> None:
        pass

    def train(self) -> jnp.ndarray:
        pass

class trainer(trainer_base):
    def __init__(self, graph, unit_index, cfg, model_type, iterate):
        super().__init__(graph, unit_index, cfg, model_type, iterate)
        

    def get_model_object(self, string: str) -> None:
        if string == 'standardised_model':
            return self.standardised_model
        elif string == 'unstandardised_model':
            return self.unstandardised_model
        elif string == 'standardisation_metrics_input':
            return self.standardisation_metrics_input
        elif string == 'standardisation_metrics_output':
            return self.standardisation_metrics_output

    def load_trainer_methods(self) -> None:
        if self.model_subclass == 'ANN':
            if self.model_class == 'regression':
                self.trainer = partial(train_ann, model_type='regressor')
            elif self.model_class == 'classification':
                self.trainer = partial(train_ann, model_type='classifier')
        elif self.model_subclass == 'GP':
            self.trainer = train_gp
        elif self.model_subclass == 'SVM':
            self.trainer = partial(train_svm, unit_index=self.unit_index, iterate=self.iterate)
        return 

    def get_data(self, successor_node: int = None) -> None:
        if (self.model_class == 'regression') and (self.model_surrogate != 'forward_evaluation_surrogate'): # TODO make sure this method exists and acts on the right component of the graph e.g. edge or node
            dataset = regression_node_data_preparation(self.graph, self.unit_index, self.cfg)
        elif (self.model_class == 'regression') and (self.model_surrogate == 'forward_evaluation_surrogate'):
            dataset = forward_evaluation_data_preparation(self.graph, self.unit_index, self.cfg, successor_node)
        elif self.model_class == 'classification': # this is only used for determining node data i..e in approximating feasibility
            data_points, labels = binary_classifier_data_preparation(self.graph, self.unit_index, self.cfg)
            if self.model_subclass == 'SVM' : dataset = (data_points, labels)
            elif self.model_subclass == 'ANN' : dataset = Dataset(X=data_points, y=labels)
        return dataset


    def train(self, node=None) -> jnp.ndarray:
        dataset = self.get_data(successor_node=node)
        self.load_trainer_methods()
        model, args = self.trainer(self.cfg, dataset, self.cfg.surrogate.num_folds) 

        if self.model_class == 'regression':
            assert len(args) == 4, "Regression model training should return 4 arguments; standardised model (i.e. model mapping from and into a standardised space), unstandardised model (i.e. model mapping from and into original data space), standardisation metrics for input and output"
            self.standardised_model, self.unstandardised_model, self.standardisation_metrics_input, self.standardisation_metrics_output = args
        elif self.model_class == 'classification':
            assert len(args) == 3, "Classification model training should return 3 arguments; standardised model, unstandardised model, standardisation metrics for input and output"
            self.standardised_model, self.unstandardised_model, self.standardisation_metrics_input = args
       
        return model