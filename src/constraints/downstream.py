import logging
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from constraints.solvers.utilities import determine_batches, create_batches   
from constraints.solvers.constructor import solver_construction
from constraints.solvers.solvers import solver_base
from constraints.casadi_evaluator import coupling_surrogate_constraint_base
from constraints.utils import standardise_model_decisions


class global_graph_upperlevel_NLP(coupling_surrogate_constraint_base):
    """
    A global graph upper-level NLP
    - solved using casadi interface with jax and IPOPT
    - parallelism is provided by multiprocessing pool
        : may be extended to jax-pmap in the future if someone develops a nice nlp solver in jax

    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs)

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node)
        # pool settings
        self.pool = pool
        # pool settings
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool == 'jax-pmap':
            raise NotImplementedError("jax-pmap is not supported for this constraint type at the moment.")
        if self.pool == 'mp-ms':
            self.evaluation_method = self.simple_evaluation 
        if self.pool == 'ray':
            self.evaluation_method = self.ray_evaluation
        
    def __call__(self):
        return self.evaluate()
    
    def evaluate(self):
        return self.ray_wrapper()
    
    def load_solver(self):
        return solver_construction(self.cfg.solvers.post_upper_level, self.cfg.solvers.post_process_solver.upper_level)


    def ray_evaluation(self, solver, max_devices, solver_processing):

        # determine the batch size
        workers, remainder = determine_batches(len(solver), max_devices)
        # split the problems
        solver_batches = create_batches(workers, solver)

        # parallelise the batch
        result_dict = {}
        evals = 0
        if self.pool == 'ray':
            for i, solve in enumerate(solver_batches):
                results = [sol(d['id'], d['data'], d['data']['cfg']) for sol, d in  solve]       
                # set off and then synchronize before moving on
                for j, result in enumerate(results):
                    r_global = solver_processing.solve_digest(*result)
                    logging.info(f"Solver status: {r_global['success']}")
                    logging.info(f"Global  objective: {r_global['objective']}")
                    logging.info(f"Global solution: {r_global['decision_variables']}")

                    result_dict[evals + j] = r_global['decision_variables']
                evals += j+1

        del solver_batches, results

        return jnp.concatenate([jnp.array([value]).reshape(1,-1) for _, value in result_dict.items()], axis=0)

    
    def ray_wrapper(self):
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """

        solver_inputs = self.evaluate_parallel(0)

        if len(list(solver_inputs[0])) > 1:
            raise NotImplementedError("Case of uncertainty in forward pass not yet implemented/optimised for parallel evaluation.")
        else:
            results = []
            solver_reshape = []
            for p in range(len(solver_inputs[0])):
                for s_i in solver_inputs.values():
                    solver_reshape.append((s_i[p].solver, s_i[p].problem_data))
            results.append(self.ray_evaluation(solver_reshape, self.cfg.max_devices, s_i[p]))

            return jnp.concatenate(results, axis=-1)

    def evaluate_parallel(self, i):
        """
        Evaluates the constraints
        """
        problem_data = self.prepare_global_problem()
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_input_i = {0: {0: {}}}
        # iterate over predecessors and evaluate the constraints
        for pred in range(1):
            for p in range(1): 
                forward_solver = solver_object.from_method(self.cfg.solvers.post_upper_level, solver_object.solver_type, problem_data[pred][p]['objective_func'], problem_data[pred][p]['bounds'], problem_data[pred][p]['constraints'])
                initial_guess = forward_solver.initial_guess()
                forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
                forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data[pred][p]['eq_rhs']
                forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data[pred][p]['eq_lhs']
                forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
                forward_solver.solver.problem_data['data']['uncertain_params'] = None
                forward_solver.solver.problem_data['id'] = i
                pred_fn_input_i[pred][p] = forward_solver.solver

        return {0: {0: forward_solver.solver}}
    

    def prepare_global_problem(self):        
        """
        Prepares the constraints surrogates and decision variables
        """
        problem_data = {0: {0: {}}}

        # prepare the forward surrogate
        x = self.graph.graph['n_design_args']  + self.graph.graph['n_aux_args'] - len(self.graph.graph['post_process_decision_indices'])

        # load the feasibility constraint surrogate
        problem_data[0][0]['constraints'] = {0: {'params':self.graph.graph["post_process_upper_classifier_serialised"], 
                                                'args': [i for i in range(x)], 
                                                'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                'model_type': self.cfg.surrogate.classifier_selection, 
                                                'g_fn': lambda x, fn: fn(x.reshape(1,-1)).reshape(-1,1)}}
        
        # load the lhs and rhs of the constraints
        problem_data[0][0]['eq_lhs'] = -jnp.ones(1,).reshape(1,1)*jnp.inf
        problem_data[0][0]['eq_rhs'] = -jnp.zeros(1,).reshape(1,1)

        # load the objective
        problem_data[0][0]['objective_func']= {'obj_fn' : -1}

        # load the standardised bounds
        # introduce bounds 
        total_ind = jnp.arange(self.graph.graph['n_design_args'] + self.graph.graph['n_aux_args'])
        dec_ind = jnp.hstack([jnp.array(self.graph.graph['post_process_decision_indices']).reshape(-1,)])
    
        # remove the lower level decision indices from the decision indices
        dec = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

        # introduce bounds 
        lb =     jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in self.graph.graph['bounds'] if bound[0] != 'None'])[dec]
        ub =     jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in self.graph.graph['bounds'] if bound[1] != 'None'])[dec]
        bounds = [lb, ub]
    
        # standardise the inputs and decisions if required
        if self.cfg.solvers.standardised:
            bounds = standardise_model_decisions(self.graph, bounds, None)

        problem_data[0][0]['bounds'] = bounds
                
        # return the forward surrogates and decision bounds
        return problem_data
    

class local_sip_base(coupling_surrogate_constraint_base):
    """
    A local single-level iterative procedure (SIP) base class
    - solved using casadi interface with jax and IPOPT
    - parallelism is provided by multiprocessing pool
        : may be extended to jax-pmap in the future if someone develops a nice nlp solver in jax

    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs, aux)

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node)
        # pool settings
        self.pool = pool
        # pool settings
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool == 'jax-pmap':
            raise NotImplementedError("jax-pmap is not supported for this constraint type at the moment.")
        if self.pool == 'mp-ms':
            self.evaluation_method = self.simple_evaluation 
        if self.pool == 'ray':
            self.evaluation_method = self.ray_evaluation

    def __call__(self, problem_data: list[jnp.ndarray] | jnp.ndarray):
        return self.evaluate(problem_data)

    def evaluate(self, problem_data: list[jnp.ndarray] | jnp.ndarray):
        return self.ray_wrapper(problem_data)

    def load_solver(self) -> solver_construction:
        return solver_construction(self.cfg.solvers.post_upper_level, self.cfg.solvers.post_process_solver.upper_level)


    def ray_evaluation(self, solver: list, max_devices: int, solver_processing):    
        solver_method = solver[0]
        solver_data = solver[1]
        results = solver_method(solver_data['id'], solver_data['data'], solver_data['data']['cfg'])
        # set off and then synchronize before moving on
        r_global = solver_processing.solve_digest(*results)
        logging.info(f"Solver status: {r_global['success']}")
        logging.info(f"Global  objective: {r_global['objective']}")
        logging.info(f"Global solution: {r_global['decision_variables']}")
        return (r_global['objective'], r_global['decision_variables'])  # return solution

    def ray_wrapper(self, problem_data: list[jnp.ndarray] | jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """
        solver_inputs = self.evaluate_parallel(0, problem_data)
        solver_reshape = (solver_inputs.solver, solver_inputs.problem_data)
        return self.ray_evaluation(solver_reshape, self.cfg.max_devices, solver_inputs)

    def evaluate_parallel(self, i: int, problem_data: list[jnp.ndarray] | jnp.ndarray) -> solver_base:
        """
        Evaluates the constraints
        """
        problem_data = self.prepare_global_problem(problem_data)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        # iterate over predecessors and evaluate the constraints:
        forward_solver = solver_object.from_method(self.cfg.solvers.post_upper_level, solver_object.solver_type, problem_data['objective_func'], problem_data['bounds'], problem_data['constraints'])
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
    - parallelism is provided by multiprocessing pool
        : may be extended to jax-pmap in the future if someone develops a nice nlp solver in jax

    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs, aux)

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node, pool)

    def prepare_global_problem(self,  problem_data: list[jnp.ndarray]) -> dict:
        """
        Prepares the constraints surrogates and decision variables
        problem_data: list of jnp arrays corresponding to the discrete
        approximations of the infinite constraint index set
        """
        problem_dict = {}

        # prepare the forward surrogate
        x = self.graph.graph['n_design_args']  + self.graph.graph['n_aux_args'] - len(self.graph.graph['post_process_decision_indices'])

        # load the feasibility constraint surrogate
        
        problem_dict['constraints'] = {0: {'params':self.graph.graph["post_process_upper_classifier_serialised"], 
                                                'args': [i for i in range(x)], 
                                                'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                'model_type': self.cfg.surrogate.classifier_selection, 
                                                'g_fn': lambda x, fn: fn(x.reshape(1,-1)).reshape(-1,1)}}
        
        # load the lhs and rhs of the constraints
        problem_dict['eq_lhs'] = -jnp.ones(1,).reshape(1,1)*jnp.inf
        problem_dict['eq_rhs'] = -jnp.zeros(1,).reshape(1,1)

        # load the objective
        problem_dict['objective_func']= {'obj_fn' : -1}

        # load the standardised bounds
        # introduce bounds 
        total_ind = jnp.arange(self.graph.graph['n_design_args'] + self.graph.graph['n_aux_args'])
        dec_ind = jnp.hstack([jnp.array(self.graph.graph['post_process_decision_indices']).reshape(-1,)])
    
        # remove the lower level decision indices from the decision indices
        dec = np.delete(total_ind, dec