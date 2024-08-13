from abc import ABC
import jax.numpy as jnp
import numpy as np
from jax.random import choice, PRNGKey

from reconstruction.samplers import sobol_sampler
from reconstruction.objects import live_set
from reconstruction.methods import construct_cartesian_product_of_live_sets


class reconstruct_base(ABC):
    def __init__(self):
        pass

    def run(self):
        pass

    def sample_live_sets(self):
        pass

    def evaluate_joint_model(self):
        pass

class reconstruction(reconstruct_base):
    def __init__(self, cfg, graph, model):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        self.live_sets_nd_proj = construct_cartesian_product_of_live_sets(graph)
        self.ls_holder = live_set(cfg, cfg.samplers.notion_of_feasibility)
        self.feasible = False

    def update_live_set(self, candidates, constraint_vals):
        """
        Check the feasibility
        :param constraint_vals: The constraint values
        :param live_set: The live set
        :param cfg: The configuration
        :return: The feasibility
        """
        # evaluate the feasibility and return those feasible candidates
        feasible_points, feasible_prob = self.ls_holder.check_live_set_membership(candidates, constraint_vals)
        # append to live set
        self.ls_holder.append_to_live_set(feasible_points, feasible_prob)
        # check if live set is complete
        return self.ls_holder.check_if_live_set_complete()    


    def run(self, mode='sobol'):
        """
        Run the joint reconstruction
        :param graph: The graph
        :param cfg: The configuration
        :param mode: The mode
        :param max_devices: The maximum number of devices
        :return: The graph
        """
        # get the livesets
        feasible = False
        ls_holder = self.ls_holder


        uncertain_params = self.get_uncertain_params()

        while not feasible:
            # sample the live sets
            live_sets_nd_proj, candidates = self.sample_live_sets(scheme=mode)
            # evaluate the joint model
            constraint_vals = self.evaluate_joint_model(candidates, uncertain_params=uncertain_params)
            constraint_vals = jnp.concatenate([g for g in constraint_vals.values()], axis=-1)
            # check feasibility
            feasible = self.update_live_set(candidates, constraint_vals)

        joint_live_set, joint_live_set_prob = ls_holder.get_live_set()
        

        return joint_live_set, joint_live_set_prob
    
    def get_uncertain_params(self):
        if self.cfg.formulation == 'probabilistic':
            param_dict = self.cfg.case_study.parameters_samples
            list_of_params = [jnp.array([p['c'] for p in param]) for param in param_dict]
            list_of_weights = [jnp.array([p['w'] for p in param]).reshape(-1) for param in param_dict]

            max_parameter_samples = self.cfg.max_uncertain_samples
            selected_params = [choice(PRNGKey(0), a, shape=(max_parameter_samples,), replace=True, p=weight, axis=0) for a, weight in zip(list_of_params, list_of_weights)]
        elif self.cfg.formulation == 'deterministic':
            selected_params = [jnp.array([param]) for param in self.cfg.case_study.parameters_best_estimate]
            
        return selected_params # sample selected parameters from    the list of parameters according to probability mass specificed by the user
    
    def sample_live_sets(self, scheme = "sobol"):
        """
        Sample from the live sets
        :param live_sets_nd_proj: The live sets
        :param cfg: The configuration
        :return: The sampled live sets
        """
        sampled_live_sets = {}
        for node, live_set in self.live_sets_nd_proj.items():
            rng = np.random.default_rng()
            # sample from the live set using bounds
            n_samples = self.cfg.samplers.ns.n_replacements
            n_l = live_set.shape[0]
            bounds = [np.zeros(1), np.ones(1)*n_l]
            if scheme == "sobol":
                # get unrounded indices
                unrounded_indices = sobol_sampler().sample_design_space(1, bounds, n_samples)
            elif scheme == "uniform":
                # get unrounded indices
                unrounded_indices = rng.uniform(bounds[0], bounds[1], (n_samples, 1))
            else:
                raise ValueError("Invalid scheme")
            
            # get rounded indices
            rnd_ind = np.round(unrounded_indices).astype(int)
            rounded_indices = np.minimum(rnd_ind, self.cfg.samplers.ns.n_live-1)
            # get shuffled live sets
            sampled_live_sets[node] = np.copy(live_set[rounded_indices[:].reshape(-1), :]).reshape(-1, live_set.shape[1])
            # shuffle live set for next round
            rng.shuffle(live_set, axis=0)
            self.live_sets_nd_proj[node] = np.copy(live_set)
            
        return self.live_sets_nd_proj, np.hstack([live_set for live_set in sampled_live_sets.values()])



    def evaluate_joint_model(self, candidates, uncertain_params):
        """
        Evaluate the joint model
        :param candidates: The candidates
        :param model: The model
        :param cfg: The configuration
        :return: The constraint values
        """
        # evaluate the joint model
        return self.model.get_constraints(candidates,  uncertain_params)
