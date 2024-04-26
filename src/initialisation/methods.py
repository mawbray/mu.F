from abc import ABC
import jax.numpy as jnp
import pandas as pd



# TODO think about ways to fit approximators to the eks data within this class.

class initialisation(ABC):
    def __init__(self, cfg, graph, network_simulator, constraint_evaluator, sampler, approximator):
        self.cfg = cfg
        self.graph = graph
        self.network_simulator = network_simulator(cfg, graph, constraint_evaluator)
        self.sampler = sampler
        self.approximator = approximator

    def run(self):
        samples = self.sample_design_space()
        uncertain_params = self.get_uncertain_params()
        constraints, eks_data = self.network_simulator.get_data(samples, uncertain_params)
        self.update_eks_data(eks_data)
        self.graph.graph["initial_forward_pass"] = pd.DataFrame({col:samples[:,i] for i,col in enumerate(self.cfg.design_space_dimensions)})
        return self.graph
    
    def update_eks_data(self, eks_data):
        # fit approximator to eks data
        for edge in eks_data.keys():
            self.graph.edges[edge[0], edge[1]]["input_data_bounds"] = self.approximator.fit(eks_data[edge], self.cfg)

        return 

    def sample_design_space(self):
        bounds = self.get_bounds()
        n_d = bounds[0].shape[0]
        return self.sampler.sample_design_space(n_d, bounds, self.cfg.n_initial_samples)
        

    def get_bounds(self):
        bounds = self.cfg.network_KS_bounds
        return self.process_bounds(self.bounds_to_dictionary(bounds))
    
    def get_uncertain_params(self):
        return self.cfg.uncertain_params # TODO confirm this is the correct APPROACH

    @staticmethod
    def process_bounds(bounds):
        """ from dictionary to array """
        # get lower and upper bounds in array form
        lower_bound = jnp.array([bounds[i][i][0] for i in bounds.keys()])
        upper_bound = jnp.array([bounds[i][i][1] for i in bounds.keys()])
        return lower_bound, upper_bound
    
    @staticmethod
    def bounds_to_dictionary(bounds):
        """ Method to construct a list of bounds for the design space"""

        bounds = {}
        index = 0
        for j, unit_bounds in enumerate(bounds):
            for i, bound in enumerate(unit_bounds):
                bounds[f'd{index}'] = {f'd{index}': [bound[0], bound[1]]}
                index += 1
            
        return bounds



def calculate_box_outer_approximation(data, config):
    """
    Calculate the box outer approximation of the given data.

    Parameters:
    data (jnp.array): The input data.
    config (object): The configuration object with a 'forward_pass.vol_scale' attribute.

    Returns:
    list: The minimum and maximum values of the box outer approximation.
    """

    # Calculate the range of the input data
    data_range = jnp.max(data, axis=0) - jnp.min(data, axis=0)

    # Calculate the increment/decrement value for the box outer approximation
    delta = config.forward_pass.vol_scale / 2 * data_range

    # Calculate the minimum and maximum values of the box outer approximation
    min_value = jnp.min(data - delta, axis=0)
    max_value = jnp.max(data + delta, axis=0)

    return [min_value.reshape(1,-1), max_value.reshape(1,-1)]