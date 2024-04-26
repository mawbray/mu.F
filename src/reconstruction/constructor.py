from abc import ABC
import numpy as np

from samplers import sobol_sample_design_space_nd
from objects import live_set
from methods import construct_cartesian_product_of_live_sets


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
    def __init__(self, cfg, graph, model, save_path):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        self.live_sets_nd_proj = construct_cartesian_product_of_live_sets(graph)
        self.ls_holder = live_set(cfg, cfg.notion_of_feasibility)
        self.feasible = False
        self.save_path = save_path

    def update_live_set(self, candidates, constraint_vals):
        """
        Check the feasibility
        :param constraint_vals: The constraint values
        :param live_set: The live set
        :param cfg: The configuration
        :return: The feasibility
        """
        # evaluate the feasibility and return those feasible candidates
        feasible_points = self.ls_holder.check_live_set_membership(candidates, constraint_vals)
        # append to live set
        self.ls_holder.append_to_live_set(feasible_points)
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

        while not feasible:
            # sample the live sets
            live_sets_nd_proj, candidates = self.sample_live_sets(scheme=mode)
            # evaluate the joint model
            constraint_vals = self.evaluate_joint_model(candidates)
            # check feasibility
            feasible = self.update_live_set(candidates, constraint_vals)

        joint_live_set = ls_holder.get_live_set()
        

        return joint_live_set


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
            n_samples = self.cfg.ns.n_replacements
            n_l = live_set.shape[0]
            bounds = [np.zeros(1), np.ones(1)*n_l]
            if scheme == "sobol":
                # get unrounded indices
                unrounded_indices = sobol_sample_design_space_nd(1, bounds, n_samples)
            elif scheme == "uniform":
                # get unrounded indices
                unrounded_indices = rng.uniform(bounds[0], bounds[1], (n_samples, 1))
            else:
                raise ValueError("Invalid scheme")
            
            # get rounded indices
            rnd_ind = np.round(unrounded_indices).astype(int)
            rounded_indices = np.minimum(rnd_ind, self.cfg.ns.n_live-1)
            # get shuffled live sets
            sampled_live_sets[node] = np.copy(live_set[rounded_indices[:].reshape(-1), :]).reshape(-1, live_set.shape[1])
            # shuffle live set for next round
            rng.shuffle(live_set, axis=0)
            self.live_sets_nd_proj[node] = np.copy(live_set)
            
        return self.live_sets_nd_proj, np.hstack([live_set for live_set in sampled_live_sets.values()])



    def evaluate_joint_model(self, candidates):
        """
        Evaluate the joint model
        :param candidates: The candidates
        :param model: The model
        :param cfg: The configuration
        :return: The constraint values
        """
        # evaluate the joint model
        return self.model.g(candidates,  np.array([1.0]))
