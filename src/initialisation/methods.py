from abc import ABC
import jax.numpy as jnp




# TODO think about ways to fit approximators to the eks data within this class.

class initialisation(ABC):
    def __init__(self, cfg, graph, network_simulator, constraint_evaluator, sampler):
        self.cfg = cfg
        self.graph = graph
        self.network_simulator = network_simulator(cfg, graph, constraint_evaluator)
        self.sampler = sampler

    def run(self):
        samples = self.sample_design_space()
        uncertain_params = self.get_uncertain_params()
        constraints, eks_data = self.network_simulator.get_data(samples, uncertain_params)
        # TODO fit approximators to eks data and load onto the graph
        return 

    def sample_design_space(self):
        bounds = self.get_bounds()
        n_d = bounds[0].shape[0]
        return self.sampler.sample_design_space(n_d, bounds, self.cfg.n_initial_samples)
        

    def get_bounds(self):
        bounds = self.cfg.network_KS_bounds
        return process_bounds(bounds_to_dictionary(bounds))
    
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
