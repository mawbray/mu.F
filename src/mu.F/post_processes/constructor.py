from abc import ABC
from time import time
from typing import Tuple, Callable
from functools import partial
import jax.numpy as jnp
from jax import jit, clear_caches
import numpy as np
import logging
from omegaconf import OmegaConf

from sipsolve.interface import SubProblemIteration
from sipsolve.discretisation import DiscretisationConfig
from sipsolve.problem_management import P1Manager, P2Manager, SubProblemInterface
from sipsolve.constants import MIN_SAMPLES
from sipsolve.constraints.utils import ConstraintEvaluatorMinMaxProjection


class post_process_base(ABC):
    def __init__(self, cfg, graph, model):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        
    def run(self):
        pass

    def train_model(self, str_: str ='post_process_'):
        if str_ == 'post_process_lower_':
            self.train_model_fn(cfg_dict=self.cfg.surrogate.post_process_lower, str_= str_)
        elif str_ == 'post_process_upper_':
            self.train_model_fn(cfg_dict=self.cfg.surrogate.post_process_upper, str_= str_)

    def train_model_fn(self, cfg_dict, str_: str ='post_process_upper_'):
        # train the surrogate model for the upper level problem
        str_root = 'classifier' if cfg_dict.model_class == 'classification' else 'regressor'
        ls_surrogate = self.training_methods(self.graph, None, self.cfg, (cfg_dict.model_class, cfg_dict.model_selection, cfg_dict.type), self.iterate, str_ + str_root + '_training')
        ls_surrogate.fit(node=None)
        if self.cfg.solvers.standardised:
            query_model = ls_surrogate.get_model('standardised_model')
        else:
            query_model = ls_surrogate.get_model('unstandardised_model')
        
        # store the trained model in the graph
        self.graph.graph[str_ + str_root] = query_model
        self.graph.graph[str_ + str_root + "_x_scalar"] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
        self.graph.graph[str_ + str_root + "_serialised"] = ls_surrogate.get_serailised_model_data()

        del ls_surrogate

        return

    def load_training_methods(self, training_methods):
        assert hasattr(training_methods, 'fit')
        self.training_methods = training_methods

    def load_solver_methods(self, solver_methods):
        self.solver_methods = solver_methods

    def _evaluate_solution(self, solution, value_fn):
        """
        Evaluate the solution of the upper-level problem
        """
        # EVALUATE THE SOLUTION
        evaluation_function = self.graph.graph['post_process_solution_evaluator']

        dataframe = evaluation_function(self.cfg, self.graph).wrap_get_constraints(solution)
        self._visualise_solution(dataframe, value_fn)
        
        return dataframe
    
    def _visualise_solution(self, solution, value_fn):
        """
        Visualise the solution of the upper-level problem
        """
        # Implement the logic to visualise the solution
        assert self.solver_methods is not None, "Solver methods must be set before visualising the solution."
        
        visualisation_function = self.graph.graph['post_process_solution_visualiser']
        visualisation_function(self.cfg, self.graph, (solution, value_fn), string='post_process_upper', path='post_process_upper').run()


class post_process_sampling_scheme(post_process_base):
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
        optima = self.optimize_nuisance_free()
        _ =  self._evaluate_solution(optima)

        return self.graph
    
    def load_feasible_infeasible(self, feasible, live_set):
        self.feasible = feasible
        self.live_set = live_set

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
        self.train_model(str_='post_process_lower_')
        evaluation_function = nuisance_constraint_evaluator(cfg=self.cfg, graph=self.graph, pool=None).evaluate
        
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
        self.graph = self.live_set.load_classification_data_to_graph(self.graph, str_='post_process_upper_')
        
        return live_set[0]
    
    def optimize_nuisance_free(self):
        """
        Second step of the post-processing: 
        Solve upper-level problem of a bilevel program. 
        """
        # get the upper level solver
        nuisance_constraint_evaluator = self.solver_methods['upper_level_solver']
        # train the model
        self.train_classification_model(str_='post_process_upper_')
        evaluation_function = nuisance_constraint_evaluator(cfg=self.cfg, graph=self.graph, pool='ray').evaluate
        # in the upper level we have no parameters to recursively evaluate, so we just solve to find an optimum.
        optimum = evaluation_function()
        logging.info(f"Local optimum found: {optimum}")

        return np.reshape(np.array(optimum).reshape(-1,)[:-1], (1,-1))

    
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


class post_process_local_sip_scheme(post_process_base):
    def __init__(self, cfg, graph, model, iterate):
        super().__init__(cfg, graph, model)
        self.feasible = None
        self.live_set = None
        self.training_methods = graph.graph['post_process_training_methods']
        self.solver_methods = None
        self.iterate = iterate
    
    def run(self):
        """
        Run the post-processing scheme using local SIP approximation.
        """
        # Implement the main logic for post-processing here
        self.relaxation_b_decisions = self.graph.graph['post_process_decision_indices'] 
        list_of_bounds = list(OmegaConf.to_container(self.cfg.case_study.KS_bounds).values())
        list_of_bounds = [[v for v in value if 'None' not in v[0]] for value in list_of_bounds]
        self.bounds_list = jnp.vstack(list_of_bounds[0] + list_of_bounds[1])
        self.relaxation_a_decisions = list(range(len(self.bounds_list)))
        for index in self.relaxation_b_decisions: self.relaxation_a_decisions.remove(index)
        assert self.relaxation_a_decisions is not None, "Decision variables must be set in the graph."
        assert self.relaxation_b_decisions is not None, "Decision variables must be set in the graph."
        assert self.solver_methods is not None, "Solver methods must be set before running the post-process."
        assert self.training_methods is not None, "Training methods must be set before running the post process."
        # train the lower level classifier
        self.train_model(str_='post_process_lower_')
        self.relaxation_a_decisions = jnp.array(self.relaxation_a_decisions)
        self.relaxation_b_decisions = jnp.array(self.relaxation_b_decisions)
        # Solve local SIP
        solution, value_fn = self.sip_approximation()
        _ =  self._evaluate_solution(solution, value_fn)
        return self.graph
    
    def _get_model(self, str_: str ='post_process_lower'):
        """
        Get the classifier from the graph
        :param str_: The string prefix
        :return: The classifier
        """
        if self.cfg.surrogate.post_process_lower.model_class == 'regression':
            assert self.graph.graph[str_ + "regressor"] is not None, "Regressor must be set in the graph."
            def regressor_fn(x):
                return self.graph.graph[str_ + "regressor"](x[0,:-1].reshape(1,-1)).reshape(1,-1) - x[0,-1].reshape(1,-1)**2
            return [regressor_fn]
        else:
            assert self.graph.graph[str_ + "classifier"] is not None, "Classifier must be set in the graph."
            return [self.graph.graph[str_ + "classifier"]]

    def sip_approximation(self):
        """
        Method to solve the SIP approximation.
        - iteratively solves relaxation a and b until convergence
        - returns the best solution found
        """
        cfg = self.cfg
        clear_caches()
        # set up discretisation scheme
        x_bounds = self.get_obj_bounds(self.relaxation_b_decisions)
        n_g = max(int(cfg.reconstruction.post_process_sip.discretisation.num_samples_per_dim * x_bounds.shape[1]), MIN_SAMPLES)
        discretisation_scheme = DiscretisationConfig(n_g, bounds=x_bounds, method=cfg.reconstruction.post_process_sip.discretisation.method)
        # set up scaling function
        def projection_fn(x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
            decisions = jnp.zeros((x.reshape(-1,).shape[0] + d.reshape(-1,).shape[0]))
            decisions = decisions.at[self.relaxation_b_decisions].set(x.reshape(-1,))
            decisions = decisions.at[self.relaxation_a_decisions].set(d.reshape(-1,))
            return decisions.reshape(1,-1)
        # get constraint system
        g_x = self._get_model(str_='post_process_lower_')
        # wrap classifier
        g_x_wrapped = [ConstraintEvaluatorMinMaxProjection(constraints=g_x)]
        # set up subproblem interface
        p1_manager = P1Manager(
                cfg.reconstruction.post_process_sip, constraints=g_x_wrapped, discretisation_scheme=discretisation_scheme, scaling_fn=projection_fn
            )
        p2_manager = P2Manager(
                cfg.reconstruction.post_process_sip, constraints=g_x_wrapped, scaling_fn=projection_fn
            )
        d_bounds = self.get_obj_bounds(self.relaxation_a_decisions)
        optimizer = SubProblemInterface(
            cfg.reconstruction.post_process_sip,
            constraint_manager=p1_manager,
            feasibility_manager=p2_manager,
            d_bounds=d_bounds,
            x_bounds=x_bounds,
            n_g=1
        )
        
        # Create the interface
        start_time = time()
        interface_instance = SubProblemIteration(cfg=cfg.reconstruction.post_process_sip, optimizer=optimizer)

        # Run the experiment
        final_optimizer, final_timing_metrics = interface_instance.create()
        logging.info(f"Experiment completed in {time() - start_time:.2f} seconds")

        # get optimizer and objects for metrics
        solution_x = final_optimizer.feasibility_manager.relaxation_data
       

        return solution_x.reshape(-1,)[:-1].reshape(1,-1), solution_x.reshape(-1,)[-1]


    def get_obj_bounds(self, decision_indices: jnp.ndarray) -> jnp.ndarray:
        return self.bounds_list[decision_indices, :].T