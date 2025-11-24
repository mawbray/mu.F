""" define any helper functions required for definition of the ODE terms"""
from abc import ABC
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def arrhenius_kinetics_fn(decision_params, uncertainty_params, Ea, A, R):
    temperature = decision_params[0] # temperature is always the first decision parameter
    return A * jnp.exp(-Ea / (R * temperature))

@jit
def arrhenius_kinetics_fn_2(decision_params, uncertainty_params, Ea, R):
    temperature = decision_params[0] # temperature is always the first decision parameter
    A = uncertainty_params
    return A * jnp.exp(-Ea / (R * temperature))
    
class RegressorData(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.inputs, self.outputs = [], []
    
    def append_to_live_set(self, x, y):
        """
        Append to the data set
        :param y: The outputs
        :param x: The input
        """
        self.inputs.append(x)
        self.outputs.append(y.reshape(-1,1))
        return
    
    def load_regression_data_to_graph(self, graph=None, str_='post_process_lower'):
        """    Get the regression data for training a regressor
        :param graph: The graph
        :param str: String identifier
        :return: The inputs and outputs for the regressor
        """
        if graph is None:
            raise ValueError("Graph must be provided to load regression data.")

        # get samples
        inputs, outputs = np.vstack(self.inputs), np.vstack(self.outputs)
        graph.graph[str_+ 'regressor_training'] = dataset(inputs, outputs)
        return graph

class dataset(ABC):
    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else np.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else np.expand_dims(y, axis=-1)
            