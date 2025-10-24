from abc import ABC
from typing import Tuple
import jax.numpy as jnp
import numpy as np
import logging

class post_process_base(ABC):
    def __init__(self, cfg, graph, model):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        

    def run(self):
        pass

    def train_classification_model(self, str_: str ='post_process_'):

        ls_surrogate = self.training_methods(self.graph, None, self.cfg, ('classification', self.cfg.surrogate.classifier_selection, 'live_set_surrogate'), self.iterate, str_ + 'classifier_training')
        ls_surrogate.fit(node=None)
        if self.cfg.solvers.standardised:
            query_model = ls_surrogate.get_model('standardised_model')
        else:
            query_model = ls_surrogate.get_model('unstandardised_model')
        
        # store the trained model in the graph
        self.graph.graph[str_ + "classifier"] = query_model
        self.graph.graph[str_ + "classifier_x_scalar"] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
        self.graph.graph[str_ + "classifier_serialised"] = ls_surrogate.get_serailised_model_data()

        logging.info(str_ + f"classifier trained with {ls_surrogate.trainer.get_model_object('standardisation_metrics_input').mean.shape} features.")

        del ls_surrogate

        return 

    def load_training_methods(self, training_methods):
        assert hasattr(training_methods, 'fit')
        self.training_methods = training_methods

    def load_solver_methods(self, solver_methods):
        self.solver_methods = solver_methods

    def _evaluate_solution(self, solution):
        """
        Evaluate the solution of the upper-level problem
        """
        # EVALUATE THE SOLUTION
        evaluation_function = self.graph.graph['post_process_solution_evaluator']

        dataframe = evaluation_function(self.cfg, self.graph).wrap_get_constraints(solution)
        self._visualise_solution(dataframe)
        
        return dataframe
    
    def _visualise_solution(self, solution):
        """
        Visualise the solution of the upper-level problem
        """
        # Implement the logic to visualise the solution
        assert self.solver_methods is not None, "Solver methods must be set before visualising the solution."
        
        visualisation_function = self.graph.graph['post_process_solution_visualiser']
        visualisation_function(self.cfg, self.graph, solution, string='post_process_upper', path='post_process_upper').run()


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
        self.train_classification_model(str_='post_process_lower_')
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
        relaxation_b_decisions = self.graph.graph['post_process_lower_decision_indices']
        relaxation_a_decisions = self.graph.graph['post_process_upper_decision_indices']
        assert relaxation_a_decisions is not None, "Decision variables must be set in the graph."
        assert relaxation_b_decisions is not None, "Decision variables must be set in the graph."
        assert self.solver_methods is not None, "Solver methods must be set before running the post-process."
        assert self.training_methods is not None, "Training methods must be set before running the post process."
        assert self.live_set is not None, "Live set must be loaded before running the post"
        # train the lower level classifier
        self.train_classification_model(str_='post_process_lower_')
        # Solve local SIP
        solution, value_fn = self.sip_approximation()
        # Evaluate the solution
        _ =  self._evaluate_solution(solution)
        return self.graph
    

    def solve_relaxation_a(self, discrete_index_set: list[jnp.ndarray]) -> Tuple[jnp.ndarray, float]:
        """
        Solve relaxation a of the SIP approximation.
        :param discrete_index_set: The discrete index set
        :return: The solution and the objective value
        """
        # Implement the logic to solve relaxation a
        assert self.solver_methods is not None, "Solver methods must be set before solving relaxation a."
        relaxation_a_solver = self.solver_methods['relaxation_a_solver']
        solution, value_fn = relaxation_a_solver(cfg=self.cfg, graph=self.graph, node=None, pool='ray', discrete_index_set=discrete_index_set, decision_indices=self.graph.graph['post_process_upper_decision_indices']).evaluate()
        
        return solution, value_fn
    
    def solve_relaxation_b(self, current_solution: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        Solve relaxation b of the SIP approximation.
        :param current_solution: The current solution from relaxation a
        :return: The solution and the objective value
        """
        # Implement the logic to solve relaxation b
        assert self.solver_methods is not None, "Solver methods must be set before solving relaxation b."
        relaxation_b_solver = self.solver_methods['relaxation_b_solver']
        solution, value_fn = relaxation_b_solver(cfg=self.cfg, graph=self.graph, pool='ray', current_solution=current_solution, decision_indices=self.graph.graph['post_process_lower_decision_indices']).evaluate()
        
        return solution, value_fn

    def relaxation_a(self, discrete_index_set: list[jnp.ndarray]) -> Tuple[jnp.ndarray, float]:
        """
        Method to solve relaxation a of the SIP approximation.
        - stores the current best solution as self.best_r1_solution
        """
        solver = self.solver_methods['upper_level_solver']
        solver = solver(cfg=self.cfg, graph=self.graph, node=None, pool='ray', constraint_type=self.cfg.reconstruction.post_process_solver.upper_level)
        solution, value_fn = self.solve_relaxation_a(solver, discrete_index_set)
        self.best_r1_solution = solution

        return solution, value_fn
    
    def relaxation_b(self, current_solution: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        Method to solve relaxation b of the SIP approximation.
        - updates the current best solution as self.best_r1_solution
        """
        solution, value_fn = self.solve_relaxation_b(current_solution)
        self.best_2_solution = solution

        return solution, value_fn

    def sip_approximation(self):
        """
        Method to solve the SIP approximation.
        - iteratively solves relaxation a and b until convergence
        - returns the best solution found
        """
        discrete_index_set = self.initialize_discrete_index_set()
        converged = False
        while not converged:
            # solve relaxation a
            solution_ve, value_fn_obj = self.relaxation_a(discrete_index_set)
            # solve relaxation b
            solution_g, value_fn_g = self.relaxation_b(solution_ve)
            # check convergence
            converged = self.check_convergence(value_fn_g)
            # update discrete index set
            if not converged:
                discrete_index_set = self.update_index_set(discrete_index_set, solution_g)

        if converged:
            logging.info("SIP approximation converged.")
            logging.info(f"Best solution found: {solution_ve} with objective value {value_fn_obj}.")
        return solution_ve, value_fn_obj

    def initialize_discrete_index_set(self) -> list[jnp.ndarray]:
        """
        Initialize the discrete index set for the SIP approximation.
        :return: The initial discrete index set
        """
        decision_bounds = self.cfg.case_study.KS_bounds

    @staticmethod
    def check_convergence(value_g: float) -> bool:
        if value_g <= 0:
            return True
        else:
            return False
    
    @staticmethod
    def update_index_set(index_set: list[jnp.ndarray], new_indices: jnp.ndarray) -> list[jnp.ndarray]:
        index_set.append(new_indices)
        return index_set

    # TODO finish implementation
    def solve_relaxation_a(problem_data: list[jnp.ndarray]) -> Tuple[jnp.ndarray, float]:
        pass

    def solve_relaxation_b(problem_data: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        pass