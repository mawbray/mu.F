from functools import partial
import logging
from typing import Tuple

import numpy as np
import jax.numpy as jnp
from networkx import Graph

from mu_F.solvers.constructor import solver_construction
from mu_F.solvers.solvers import solver_base
from mu_F.constraints.casadi_evaluator import coupling_surrogate_constraint_base
from mu_F.constraints.utils import standardise_model_decisions, mask_classifier


class global_graph_solver_base(coupling_surrogate_constraint_base):
    """
    A global graph solver base class
    - solved using casadi interface with jax and IPOPT

    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate()
    """
    def __init__(self, cfg: dict, graph: Graph, node: int):
        super().__init__(cfg, graph, node)
        
    def __call__(self):
        return self.evaluate()
    
    def evaluate(self):
        return self.wrapper()
    
    def load_solver(self) -> solver_construction:
        """
        Loads the solver factory instance.
        
        Returns:
            solver_construction instance (not yet configured with objective/constraints)
        """
        from mu_F.solvers.constructor import SolverType
        # Convert string config to SolverType enum
        solver_type = SolverType(self.cfg.solvers.post_process_solver.upper_level)
        return solver_construction(self.cfg.solvers.post_upper_level, solver_type)

    def evaluation(self, solver: list, get_results: callable):    
        """
        Evaluates the constraints in parallel using ray.
        :solver: a list of solver instances to be evaluated in parallel
        :solver_processing: a solver processing instance to handle the results
        """

        solver_method = solver[0]
        solver_data = solver[1]
        results = solver_method(solver_data['id'], solver_data['data'], solver_data['data']['cfg'])
        # set off and then synchronize before moving on
        r_global = get_results(*results)
        logging.info(f"Solver status: {r_global['success']}")
        logging.info(f"Global  objective: {r_global['objective']}")
        logging.info(f"Global solution: {r_global['decision_variables']}")
        return (r_global['objective'], r_global['decision_variables'])  # return solution
    
    def wrapper(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.

        """
        solver_inputs = self.evaluate_parallel(0)
        solver_reshape = (solver_inputs.solver, solver_inputs.problem_data)
        return self.evaluation(solver_reshape, solver_inputs)

    def evaluate_parallel(self, i):
        """
        generates the solver instance
        """
        # prepare the problem data
        problem_data = self.prepare_global_problem()
        # solver type has been defined elsewhere in the case study/graph construction. 
        solver_object = self.load_solver()  
        # instantiate the solver and major functions
        forward_solver = solver_object.from_method(
            self.cfg.solvers.post_upper_level,
            solver_object.solver_type.value,  # Convert enum to string
            problem_data['objective_func'],
            problem_data['bounds'],
            problem_data['constraints']
        )
        # finish problem set up
        initial_guess = forward_solver.initial_guess()
        forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
        forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data['eq_rhs']
        forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data['eq_lhs']
        forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
        forward_solver.solver.problem_data['data']['uncertain_params'] = None
        forward_solver.solver.problem_data['id'] = i

        return forward_solver.solver
    
    def prepare_global_problem(self) -> dict:
        raise NotImplementedError("This method should be implemented in a subclass.")


class global_graph_upperlevel_NLP(global_graph_solver_base):
    """
    A global graph upper-level NLP
    - solved using casadi interface with jax and IPOPT
    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate()

    """
    def __init__(self, cfg: dict, graph: Graph, node: int):
        super().__init__(cfg, graph, node)

    def prepare_global_problem(self):
        """
        Prepares the constraints surrogates and decision variables
        """
        problem_data = {}
        # prepare the forward surrogate
        x = self.graph.graph['n_design_args']  + self.graph.graph['n_aux_args'] - len(self.graph.graph['post_process_decision_indices'])

        # load the feasibility constraint surrogate
        problem_data['constraints'] = {0: {'params':self.graph.graph["post_process_upper_classifier_serialised"], 
                                                'args': [i for i in range(x)], 
                                                'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                'model_type': self.cfg.surrogate.classifier_selection, 
                                                'g_fn': lambda x, fn: fn(x.reshape(1,-1)).reshape(-1,1)}}
        
        # load the lhs and rhs of the constraints
        problem_data['eq_lhs'] = -jnp.ones(1,).reshape(1,1)*jnp.inf
        problem_data['eq_rhs'] = -jnp.zeros(1,).reshape(1,1)

        # load the objective
        # based on logic in :solvers.functions.ray_casadi_multi_start
        # this defines the index of the decision variable to be minimised
        problem_data['objective_func'] = {'obj_fn' : -1} 

        # introduce bounds 
        problem_data['bounds'] = self._get_bounds(self.graph, self.cfg)

        # return the forward surrogates and decision bounds
        return problem_data
    
    @staticmethod
    def _get_bounds(graph: Graph, cfg: dict) -> list[jnp.ndarray]:
        """
        Get the bounds for the decision variables
        """
        # determine the indices of the decision variables
        total_ind = jnp.arange(graph.graph['n_design_args'] + graph.graph['n_aux_args'])
        dec_ind = jnp.hstack([jnp.array(graph.graph['post_process_decision_indices']).reshape(-1,)])
    
        # remove the lower level decision indices from the decision indices
        dec = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

        # introduce bounds
        lb = jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in graph.graph['bounds'] if bound[0] != 'None'])
        ub = jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in graph.graph['bounds'] if bound[1] != 'None'])
        bounds = [lb, ub] if not cfg.solvers.standardised else standardise_model_decisions(graph, [lb, ub], None)

        return [bounds[0][dec], bounds[1][dec]]

class local_sip_base(coupling_surrogate_constraint_base):
    """
    A local single-level iterative procedure (SIP) base class
    - solved using casadi interface with jax and IPOPT
    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs, aux)
    """
    def __init__(self, cfg: dict, graph: Graph, node: int):
        super().__init__(cfg, graph, node)

    def __call__(self, problem_data: list[jnp.ndarray] | jnp.ndarray):
        return self.evaluate(problem_data)

    def evaluate(self, problem_data: list[jnp.ndarray] | jnp.ndarray):
        return self.wrapper(problem_data)

    def load_solver(self) -> solver_construction:
        """
        Loads the solver factory instance.
        
        Returns:
            solver_construction instance (not yet configured with objective/constraints)
        """
        from mu_F.solvers.constructor import SolverType
        # Convert string config to SolverType enum
        solver_type = SolverType(self.cfg.solvers.post_process_solver.upper_level)
        return solver_construction(self.cfg.solvers.post_upper_level, solver_type)

    def wrapper(self, problem_data: list[jnp.ndarray] | jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """
        solver_inputs = self.evaluate_parallel(0, problem_data)
        solver_reshape = (solver_inputs.solver, solver_inputs.problem_data)
        return self.evaluation(solver_reshape, self.cfg.max_devices, solver_inputs)
    
    def evaluation(self, solver: list, get_results: callable):    
        """
        Evaluates the constraints in parallel using ray.
        :solver: a list of solver instances to be evaluated in parallel
        :solver_processing: a solver processing instance to handle the results
        """

        solver_method = solver[0]
        solver_data = solver[1]
        results = solver_method(solver_data['id'], solver_data['data'], solver_data['data']['cfg'])
        # set off and then synchronize before moving on
        r_global = get_results(*results)
        logging.info(f"Solver status: {r_global['success']}")
        logging.info(f"Global  objective: {r_global['objective']}")
        logging.info(f"Global solution: {r_global['decision_variables']}")
        return (r_global['objective'], r_global['decision_variables'])  # return solution

    def evaluate_parallel(self, i: int, problem_data: list[jnp.ndarray] | jnp.ndarray) -> solver_base:
        """
        generates the solver instance
        """
        problem_data = self.prepare_global_problem(problem_data)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        # iterate over predecessors and evaluate the constraints:
        forward_solver = solver_object.from_method(
            self.cfg.solvers.post_upper_level,
            solver_object.solver_type.value,  # Convert enum to string
            problem_data['objective_func'],
            problem_data['bounds'],
            problem_data['constraints']
        )
        initial_guess = forward_solver.initial_guess()
        forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
        forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data['eq_rhs']
        forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data['eq_lhs']
        forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
        forward_solver.solver.problem_data['data']['uncertain_params'] = None
        forward_solver.solver.problem_data['id'] = i
        return forward_solver.solver

    def prepare_global_problem(self,  problem_data: list[jnp.ndarray] | jnp.ndarray) -> dict:
        raise NotImplementedError("This method should be implemented in a subclass.")
    

class local_sip_discrete_approximation(local_sip_base):
    """
    A local single-level iterative procedure (SIP) with discrete approximation
    - solved using casadi interface with jax and IPOPT
    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs, aux)

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg: dict, graph: Graph, node: int, pool: str):
        super().__init__(cfg, graph, node, pool)

    def prepare_global_problem(self,  problem_data: list[jnp.ndarray]) -> dict:
        """
        Prepares the constraints surrogates and decision variables
        problem_data: list of jnp arrays corresponding to the discrete
        approximations of the infinite constraint index set
        """
        problem_dict = {}

        # get the number of decision variables
        n_decisions = self.graph.graph['n_design_args']  + self.graph.graph['n_aux_args'] - len(self.graph.graph['post_process_decision_indices'])

        # load the feasibility constraint surrogates
        constraint_funs = self._get_feasibility_constraint_index_set(self.graph, problem_data)

        for i, constraint_func in enumerate(constraint_funs):
            problem_dict['constraints'][i] = {'params': self.graph.graph["post_process_lower_classifier_serialised"],
                                                'args': [k for k in range(n_decisions)],
                                                'model_class': 'classification', 'model_surrogate': 'live_set_surrogate',
                                                'model_type': self.cfg.surrogate.classifier_selection,
                                                'g_fn': constraint_func}

        # load the lhs and rhs of the constraints
        problem_dict['eq_lhs'] = -jnp.ones(len(constraint_funs),).reshape(-1,1)*jnp.inf
        problem_dict['eq_rhs'] = -jnp.zeros(len(constraint_funs),).reshape(-1,1)

        # load the objective
        # based on logic in :solvers.functions.ray_casadi_multi_start
        # this defines the index of the decision variable to be minimised
        problem_dict['objective_func']= {'obj_fn' : -1}

        # introduce bounds
        problem_dict['bounds'] = self._get_bounds(self.graph, self.cfg)

        return problem_dict

    @staticmethod
    def _get_feasibility_constraint_index_set(graph: Graph, problem_data: list[jnp.ndarray] ) -> list[partial]:
        """
        Get the set of feasibility constraints.
        :graph: the graph object
        :problem_data: elements of the infinite constraint index set
        :return: a list of partial functions which will define the constraints
        """
        ndim= graph.graph['n_design_args'] + graph.graph['n_aux_args']
        total_ind = jnp.arange(ndim)  # total number of inputs to the classifier
        dec_ind = jnp.hstack([jnp.array(graph.graph['post_process_decision_indices']).reshape(-1,)])
        fix_ind = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

        return [lambda x, fn: mask_classifier(fn, ndim, fix_ind, jnp.empty((0,0)))(x, y) for y in problem_data]

    @staticmethod
    def _get_bounds(graph: Graph, cfg: dict) -> list[jnp.ndarray]:
        """
        Get the bounds for the decision variables
        """
        # determine the indices of the decision variables
        total_ind = jnp.arange(graph.graph['n_design_args'] + graph.graph['n_aux_args'])
        dec_ind = jnp.hstack([jnp.array(graph.graph['post_process_decision_indices']).reshape(-1,)])
    
        # remove the lower level decision indices from the decision indices
        dec = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

        # introduce bounds 
        lb =     jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in graph.graph['bounds'] if bound[0] != 'None'])
        ub =     jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in graph.graph['bounds'] if bound[1] != 'None'])
        bounds = [lb, ub] if not cfg.solvers.standardised else standardise_model_decisions(graph, [lb, ub], None)

        return [bounds[0][dec], bounds[1][dec]]

class local_sip_feasibility_approximation(local_sip_base):
    """
    A local single-level iterative procedure (SIP) with feasibility approximation
    - solved using casadi interface with jax and IPOPT
    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs, aux)
    """

    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node, pool)

    def prepare_global_problem(self,  problem_data: jnp.ndarray) -> dict:
        """
        Prepares the global problem formulation.
        :param problem_data: The problem data to be used in the formulation.
        :return: A dictionary containing the global problem formulation.
        """
        problem_dict = {}
        # get the number of decision variables
        n_decisions = len(self.graph.graph['post_process_decision_indices'])

        # Load the feasibility constraint surrogates
        objective_fun = self._get_feasibility_objective_function(self.graph, problem_data)


        # Load the objective
        # Based on logic in :solvers.functions.ray_casadi_multi_start
        problem_data['objective_func'] = {'f0': {'params': self.graph.graph["post_process_lower_classifier_serialised"], 
                                                            'args': [i for i in range(n_decisions)],
                                                            'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                            'model_type': self.cfg.surrogate.classifier_selection},
                                                            'obj_fn': objective_fun}
        
        # introduce bounds
        problem_dict['bounds'] = self._get_bounds(self.graph, self.cfg)

        return problem_dict

    @staticmethod
    def _get_feasibility_objective_function(graph: Graph, problem_data: list[jnp.ndarray] ) -> partial:
        """
        Get the feasibility objective.
        :graph: the graph object
        :problem_data: the optimal solution yielded from :local_sip_discrete_approximation.call()
        :return: a partial function which will define the objective of the feasibility problem
        """
        ndim= graph.graph['n_design_args'] + graph.graph['n_aux_args']
        fix_ind = jnp.hstack([jnp.array(graph.graph['post_process_decision_indices']).reshape(-1,)])

        return lambda x, fn: mask_classifier(fn, ndim, fix_ind, jnp.empty((0,0)))(x, problem_data)

    @staticmethod
    def _get_bounds(graph: Graph, cfg: dict) -> list[jnp.ndarray]:
        """
        Get the bounds for the decision variables
        :graph: the graph object
        :cfg: the configuration dictionary
        :return: a list containing the lower and upper bounds for the decision variables
        """
        # determine the indices of the decision variables
        dec_ind = jnp.hstack([jnp.array(graph.graph['post_process_decision_indices']).reshape(-1,)])

        # introduce bounds
        lb = jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in graph.graph['bounds'] if bound[0] != 'None'])
        ub = jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in graph.graph['bounds'] if bound[1] != 'None'])
        bounds = [lb, ub] if not cfg.solvers.standardised else standardise_model_decisions(graph, [lb, ub], None)

        return [bounds[0][dec_ind], bounds[1][dec_ind]]
