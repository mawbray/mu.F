from abc import ABC
import jax.numpy as jnp
import numpy as np
from jax.random import choice, PRNGKey
from functools import partial
import logging
import ray

from hydra.utils import get_original_cwd

class post_process_base(ABC):
    def __init__(self, cfg, graph, model):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        

    def run(self):
        pass

    def load_feasible_infeasible(self, feasible, live_set):
        self.feasible = feasible
        self.live_set = live_set

    def load_training_methods(self, training_methods):
        assert hasattr(training_methods, 'fit')
        self.training_methods = training_methods

    def load_solver_methods(self, solver_methods):
        self.solver_methods = solver_methods

    def solve_for_nuisnace_parameters(self, decision_variables: list[int], bounds: list[list[float]], notion: str= 'max') -> list:
        """
        Solve for nuisance parameters
        :param decision_variables: The decision variables
        :param bounds: The bounds for the decision variables
        :param notion: The notion of the problem (max or min)
        :return: The solution for the having factored out nuisance parameters
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


class post_process(post_process_base):
    def __init__(self, cfg, graph, model, iterate):
        super().__init__(cfg, graph, model)
        self.feasible = None
        self.live_set = None
        self.training_methods = graph.graph['post_process_training_methods']
        self.solver_methods = None
        self.sampler = None
        self.iterate = iterate
    
    def run(self):
        # Implement the main logic for post-processing here
        decision_vars = self.graph.graph['post_process_decision_indices'] 
        assert decision_vars is not None, "Decision variables must be set in the graph."
        assert self.solver_methods is not None, "Solver methods must be set before running the post-process."
        assert self.sampler is not None, "Sampler must be set before running the post-process."
        assert self.training_methods is not None, "Training methods must be set before running the post process."
        assert self.live_set is not None, "Live set must be loaded before running the post"
        # TODO check live set, sampler and solver methods are set correctly
        # Solve for nuisance parameters
        live_set = self.solve_for_nuisance_parameters(decision_vars)
        # Update the graph with the live set
        self.graph.graph['post_processed_live_set'] = live_set
        # Solve the upper-level problem
        self.optimize_nuisance_free()

        return self.graph

    def load_fresh_live_set(self, live_set):
        assert hasattr(live_set, 'get_live_set'), "Live set must have a method 'get_live_set'."
        assert hasattr(live_set, 'live_set_len'), "Live set must have a method 'live_set_len'."
        assert hasattr(live_set, 'check_if_live_set_complete'), "Live set must have a method 'check_if_live_set_complete'."
        assert hasattr(live_set, 'evaluate_feasibility'), "Live set must have a method 'evaluate_feasibility'."
        assert hasattr(live_set, 'check_live_set_membership'), "Live set must have a method 'check_live_set_membership'."
        assert hasattr(live_set, 'append_to_live_set'), "Live set must have a method 'append_to_live_set'."
        self.live_set = live_set

    def update_live_set(self, candidates, constraint_vals):
        """
        Check the feasibility
        :param constraint_vals: The constraint values
        :param live_set: The live set
        :param cfg: The configuration
        :return: The feasibility boolean 
        """
        # evaluate the feasibility and return those feasible candidates
        feasible_points, feasible_prob = self.live_set.check_live_set_membership(candidates, constraint_vals)
        # append to live set
        self.live_set.append_to_live_set(feasible_points, feasible_prob)
        # check if live set is complete
        return self.live_set.check_if_live_set_complete()
    
    def train_classification_model(self):

        ls_surrogate = self.training_methods(self.graph, None, self.cfg, ('classification', self.cfg.surrogate.classifier_selection, 'live_set_surrogate'), self.iterate)
        ls_surrogate.fit(node=None)
        if self.cfg.solvers.standardised:
            query_model = ls_surrogate.get_model('standardised_model')
        else:
            query_model = ls_surrogate.get_model('unstandardised_model')
        
        # store the trained model in the graph
        self.graph.graph["post_process_classifier"] = query_model
        self.graph.graph['post_process_classifier_x_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
        self.graph.graph['post_process_classifier_serialised'] = ls_surrogate.get_serailised_model_data()

        logging.info(f"Post-process classifier trained with {ls_surrogate.trainer.get_model_object('standardisation_metrics_input').mean.shape} features.")

        del ls_surrogate

        return 

    def solve_for_nuisance_parameters(self, decision_variables: list[int]) -> list:
        """
        Solve the lower level problem of a bilevel program.
        :param decision_variables: The decision variables to be factored out
        :return: The solution for the nuisance parameters"""
        # Implement the logic to solve for nuisance parameters
        assert self.solver_methods is not None, "Solver methods must be set before solving for nuisance parameters."
        assert self.sampler is not None, "Sampler must be set before solving for nuisance parameters."

        # the evaluator takes the decision variables, bounds and the feasibility function and evaluates feasibility of query points.
        nuisance_constraint_evaluator = self.solver_methods['lower_level_solver']
        # train the model
        self.train_classification_model()
        evaluation_function = nuisance_constraint_evaluator(cfg=self.cfg, graph=self.graph, node=None, pool=None, constraint_type=self.cfg.reconstruction.post_process_solver.lower_level).evaluate
        
        boolean = False
        while not boolean:
            query_points  = self.sampler()
            query_points  = self.filter_decision_variables(decision_variables, query_points)
            
            feasibility_values = evaluation_function(jnp.expand_dims(query_points, axis=1), jnp.empty((query_points.shape[0], 1, 0)))
            # repeating evaluation of points
            boolean = self.update_live_set(query_points, feasibility_values)
        logging.info(f'Live set complete with {self.live_set.live_set_len()} points of {query_points.shape[1]} dimensions.')
        # return the live set
        live_set = self.live_set.get_live_set()
        # store the data generated in characterizing the live set on the graph
        self.graph = self.live_set.load_classification_data_to_graph(self.graph, str='post_process_classifier_training')
        
        
        return live_set[0]
    
    def optimize_nuisance_free(self):
        """
        Second step of the post-processing: 
        Solve upper-level problem of a bilevel program. 
        """
        # get the upper level solver
        nuisance_constraint_evaluator = self.solver_methods['upper_level_solver']
        # train the model
        self.train_classification_model()
        evaluation_function = nuisance_constraint_evaluator(cfg=self.cfg, graph=self.graph, node=None, pool='ray', constraint_type=self.cfg.reconstruction.post_process_solver.upper_level).evaluate
        # in the upper level we have no parameters to recursively evaluate, so we just solve to find an optimum.
        optimum = evaluation_function()
        logging.info(f"Local optimum found: {optimum}")

        return optimum

    def filter_decision_variables(self, decision_variables: list[int], query_points: jnp.ndarray) -> list[int]:
        """
        Filter decision variables from array of query points
        :param decision_variables: The decision variables to be filtered
        :param query_points: The array of query points
        :return: Filtered query points
        """
        indices = jnp.arange(query_points.shape[-1])
        fixed_indices = indices[~jnp.isin(indices, jnp.array(decision_variables))]
        query_points_wo_decisions = query_points[..., fixed_indices]
        return query_points_wo_decisions
