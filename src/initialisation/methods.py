from abc import ABC
import jax.numpy as jnp
import pandas as pd
from jax.random import PRNGKey, choice



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
        self.graph.graph["initial_forward_pass"] = pd.DataFrame({col:samples[:,i] for i,col in enumerate(self.cfg.case_study.design_space_dimensions)})
        return self.graph
    
    def update_eks_data(self, eks_data):
        # fit approximator to eks data
        for edge in eks_data.keys():
            self.graph.edges[edge[0], edge[1]]["input_data_bounds"] = self.approximator(eks_data[edge], self.cfg)

        return 

    def sample_design_space(self):
        bounds = self.get_bounds()
        n_d = len(bounds[0])
        return self.sampler.sample_design_space(n_d, bounds, self.cfg.init.sobol_samples)
        

    def get_bounds(self):
        bounds = self.cfg.case_study.KS_bounds
        return self.process_bounds(self.bounds_to_dictionary(bounds))
    
    def get_uncertain_params(self):
        param_dict = self.cfg.case_study.parameters_samples
        list_of_params = [jnp.array([p['c'] for p in param]) for param in param_dict]
        list_of_weights = [jnp.array([p['w'] for p in param]).reshape(-1) for param in param_dict]

        max_parameter_samples = self.cfg.init.max_uncertain_samples
        selected_params = [choice(PRNGKey(0), a, shape=(max_parameter_samples,), replace=True, p=weight, axis=0) for a, weight in zip(list_of_params, list_of_weights)]

        return selected_params # sample selected parameters from the list of parameters according to probability mass specificed by the user

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

        bounds_ = {}
        index = 0
        for j, unit_bounds in enumerate(bounds):
            for i, bound in enumerate(unit_bounds):
                bounds_[f'd{index}'] = {f'd{index}': [bound[0], bound[1]]}
                index += 1
            
        return bounds_


