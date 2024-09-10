from abc import ABC
import os
from typing import Iterable, Callable, List
from omegaconf import DictConfig
import logging
from hydra.utils import get_original_cwd

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, pmap, devices
from functools import partial
from scipy.stats import beta
import jax.scipy.stats as jscp_stats
from copy import copy, deepcopy
import ray
#ray.init()

from constraints.solvers.functions import multi_start_solve_bounds_nonlinear_program, generate_initial_guess
from constraints.solvers.utilities import worker_function, parallelise_ray_batch, determine_batches, create_batches   
from constraints.solvers.constructor import solver_construction

class constraint_evaluator_base(ABC):
    def __init__(self, cfg, graph, node):
        self.cfg = cfg.copy()
        self.graph = graph.copy()
        self.node = node

        # shaping function to return to sampler. (depends on the way constraints are defined by the user)
        if cfg.samplers.notion_of_feasibility == 'positive': # i.e. feasible is g(x)>=0 as default the constraints are defined as g(x)>=0   
            self.shaping_function = lambda x: x
        elif cfg.samplers.notion_of_feasibility == 'negative': # i.e. feasible is g(x)<=0
            self.shaping_function = lambda x: -x
        else:
            raise ValueError("Invalid notion of feasibility.")

    def evaluate(self, dynamics_profile):
        raise NotImplementedError
    
    def load_unit_constraints(self):
        raise NotImplementedError
    

class process_constraint_evaluator(constraint_evaluator_base):
    """
    Means to simply evaluate the process constraints imposed on a unit.
    """

    def __init__(self, cfg, graph, node, pool=None):
        """
        Initializes
        """
        super().__init__(cfg, graph, node)
        if cfg.case_study.vmap_evaluations:
            self.vmap_evaluation()

    def __call__(self, design, inputs, dynamics_profile, aux):
        return self.evaluate(design, inputs, aux, dynamics_profile )

    def evaluate(self, design_args, input_args, aux_args, dynamics_profile):
        """
        Evaluates the constraints
        """
        constraints = self.load_unit_constraints()
        #dargs = jnp.repeat(jnp.expand_dims(design_args,axis=1), dynamics_profile.shape[1], axis=1)
        #aux_args = jnp.repeat(jnp.expand_dims(aux_args,axis=1), dynamics_profile.shape[1], axis=1)
        
        if len(constraints) > 0: 
            constraint_holder = []
            for cons_fn in constraints: # iterate over the constraints that were previously loaded as a dictionary on to the graph
                g = cons_fn(dynamics_profile) # positive constraint value means constraint is satisfied
                if g.ndim < 2: g = g.reshape(-1, 1)
                if g.ndim < 3: g = jnp.expand_dims(g, axis=-1)
                constraint_holder.append(g)
            return jnp.concatenate(constraint_holder, axis=-1)
        else:
            return self.shaping_function(jnp.ones(dynamics_profile.shape)) # return None if no unit level constraints are imposed.
        
    def load_unit_constraints(self):
        """
        Loads the constraints from the graph 
        """
        return list(self.graph.nodes[self.node]['constraints'].copy())
    
    def vmap_evaluation(self):
        """
        Vectorizes the the constraints and then loads them back onto the graph
        """
        # get constraints from the graph
        constraints = self.graph.nodes[self.node]['constraints']
        # vectorize each constraint
        cons = [jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model), in_axes=(0), out_axes=0)), in_axes=(1), out_axes=1)) for constraint in constraints]
        # load the vectorized constraints back onto the graph
        self.graph.nodes[self.node]['constraints'] = cons

        return 

class coupling_surrogate_constraint_base(constraint_evaluator_base):
    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def prepare_forward_surrogate(self, inputs):
        """
        Prepares the forward constraints surrogates and decision variables
        """
        raise NotImplementedError("Method not implemented")
    
    def load_solver(self):
        """
        Loads the solver
        """
        raise NotImplementedError("Method not implemented")

    def standardise_inputs(self, inputs, in_node):
        """
        Standardises the inputs
        """
        raise NotImplementedError("Method not implemented")
    
    def standardise_model_decisions(self, decisions, in_node):
        """
        Standardises the decisions
        """
        raise NotImplementedError("Method not implemented")
    
    def evaluate(self, inputs):
        """
        Evaluates the constraints
        """
        raise NotImplementedError("Method not implemented")
    
    def mp_evaluation(
        self,
        solver: Iterable[Callable],
        initial_guess: Iterable[np.ndarray],
        max_devices: int,
    ):
        """
        Parallel evaluation of constrained NLPs using multiprocessing pool.
        :param objectives: List of objective functions
        :param constraints: List of constraint functions
        :param decision_bounds: List of decision bounds
        :param cfg: Configuration
        """

        # determine the batch size
        workers, remainder = determine_batches(len(solver), max_devices)
        # split the objectives, constraints and bounds into batches
        solver_batches = create_batches(workers, solver)
        initial_guess_batches = create_batches(workers, initial_guess)
                
        # parallelise the batch
        result_dict = {}
        evals = 0
        for i, (solver, init_guess, device) in enumerate(zip(solver_batches, initial_guess_batches, workers)):
            results = parallelise_ray_batch(solver, init_guess)
            for j, result in enumerate(results):
                result_dict[evals + j] = result
            evals += len(results)

        return result_dict
    
    def simple_evaluation(self, solver, initial_guess):
        return solver(initial_guess)



class forward_constraint_evaluator(coupling_surrogate_constraint_base):
    """
    A forward surrogate constraint
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
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool == 'jax-pmap':
            raise NotImplementedError("jax-pmap is not supported for this constraint type at the moment.")
        if self.pool == 'mp-ms':
            self.evaluation_method = self.simple_evaluation 
        if self.pool == 'ray':
            self.evaluation_method = self.ray_evaluation
        
    def __call__(self, inputs, aux):
        return self.evaluate(inputs, aux)
    
    def ray_evaluation(self, solver, max_devices, solver_processing):

        # determine the batch size
        workers, remainder = determine_batches(len(solver), max_devices)
        # split the problems
        solver_batches = create_batches(workers, solver)

        # parallelise the batch
        result_dict = {}

        evals = 0
        
        for i, solve in enumerate(solver_batches):
            results = ray.get([sol.remote(d['id'], d['data'], d['data']['cfg']) for sol, d in  solve]) # set off and then synchronize before moving on
            for j, result in enumerate(results):
                result_dict[evals + j] = solver_processing.solve_digest(*result)['objective']
            evals += j+1

        del solver_batches, results
    

        return jnp.concatenate([jnp.array([value]).reshape(1,-1) for _, value in result_dict.items()], axis=0)



    def get_predecessors_inputs(self, inputs, aux):
        """
        Gets the inputs from the predecessors
        """
        pred_inputs, pred_aux = {}, {}
        for pred in self.graph.predecessors(self.node):
            input_indices = self.graph.edges[pred, self.node]['input_indices']
            aux_indices = self.graph.edges[pred, self.node]['auxiliary_indices']
            pred_inputs[pred]= inputs[:,input_indices]
            pred_aux[pred] = aux[:,aux_indices]

        return pred_inputs, pred_aux


    def load_solver(self):
        """
        Loads the solver
        """
        return solver_construction(self.cfg.solvers.forward_coupling, self.cfg.solvers.forward_coupling_solver)
    
    def evaluate(self, inputs, aux):
        """
        Evaluates the constraints by iterating sequentially over the design and the uncertain params
        inputs: samples in the extended design space
        outputs: the constraint evaluations
        """
        if self.pool == 'mp-ms':
            return self.serial_wrapper(inputs, aux)
        elif self.pool == 'ray':
            return self.ray_wrapper(inputs, aux)
            
    def parallel_wrapper(self, inputs):
        raise NotImplementedError("Method not implemented")
    
    def ray_wrapper(self, inputs, aux):
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """

        solver_inputs = []
        for i in range(inputs.shape[0]):
            solver_inputs.append(self.evaluate_parallel(i, inputs[i,:].reshape(1,-1), aux[i,:].reshape(1,-1)))        

        if len(list(solver_inputs[0].values())) > 1:
            raise NotImplementedError("Case of uncertainty in forward pass not yet implemented/optimised for parallel evaluation.")
        else:
            results = []
            for pred in self.graph.predecessors(self.node):
                solver_reshape = []
                for p in range(len(solver_inputs[0][pred])):
                    for s_i in solver_inputs:
                        solver_reshape.append((s_i[pred][p].solver, s_i[pred][p].problem_data))
                results.append(self.ray_evaluation(solver_reshape, self.cfg.max_devices, s_i[pred][p]))


            return jnp.concatenate(results, axis=-1)

        

    def evaluate_parallel(self, i, inputs, auxs):

        """
        Evaluates the constraints
        """
        problem_data = self.prepare_forward_problem_ray(inputs, auxs)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_input_i = {pred: {} for pred in self.graph.predecessors(self.node)}

        solved_successful = 0
        problems = sum([len(problem_data[pred]) for pred in self.graph.predecessors(self.node)])

        # iterate over predecessors and evaluate the constraints
        for pred in self.graph.predecessors(self.node):
            for p in range(len(problem_data[pred])): 
                forward_solver = solver_object.from_method(self.cfg.solvers.forward_coupling, solver_object.solver_type, problem_data[pred][p]['objective_func'], problem_data[pred][p]['bounds'], problem_data[pred][p]['equality_constraints'])
                initial_guess = forward_solver.initial_guess()
                forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
                forward_solver.solver.problem_data['data']['eqc_rhs'] = problem_data[pred][p]['eqc_rhs']
                forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg).copy()
                forward_solver.solver.problem_data['data']['uncertain_params'] = None
                forward_solver.solver.problem_data['id'] = i
                pred_fn_input_i[pred][p] = forward_solver.solver

        return pred_fn_input_i

    def serial_wrapper(self, inputs, aux):
        results = []
        for i in range(inputs.shape[0]):
            results.append(self.evaluate_serial(inputs[i,:].reshape(1,-1), aux=aux[i,:].reshape(1,-1)))
        

        return jnp.vstack(results)

            
    def evaluate_serial(self, inputs, aux):
        """
        Evaluates the constraints
        """
        objective, constraints, bounds = self.prepare_forward_problem(inputs, aux)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_evaluations = {pred: {} for pred in self.graph.predecessors(self.node)}

        solved_successful = 0
        problems = sum([len(constraints[pred]) for pred in self.graph.predecessors(self.node)])

        # iterate over predecessors and evaluate the constraints
        for pred in self.graph.predecessors(self.node):
            for p in range(len(constraints[pred])): 
                forward_solver = solver_object.from_method(self.cfg.solvers.forward_coupling, solver_object.solver_type, objective[pred][p], bounds[pred][p], constraints[pred][p])
                initial_guess = forward_solver.initial_guess()
                pred_fn_evaluations[pred][p] = self.evaluation_method(forward_solver, initial_guess)

                # Update the number of NLP SOLVED TO CONVERGENCE
                solved_successful += pred_fn_evaluations[pred][p]['success']
                

                del forward_solver, initial_guess

                if (np.array(pred_fn_evaluations[pred][p]['objective']) <= 0) and (pred_fn_evaluations[pred][p]['success']): break
                

        # reshape the evaluations and just get information about the constraints.
        fn_evaluations = [min([jnp.array(pred_fn_evaluations[pred][p]['objective']) for p in pred_fn_evaluations[pred].keys()]) for pred in self.graph.predecessors(self.node)]

        del pred_fn_evaluations, objective, constraints, bounds

        if solved_successful != problems: logging.info(f"forward nlp solved to convergence: {solved_successful} out of {problems}")

    
        return self.shaping_function(jnp.concatenate(fn_evaluations, axis=-1))

    def get_predecessors_uncertain(self):
        """
        Gets the uncertain parameters from the predecessors dynamics
        """
        pred_uncertain_params = {}
        for pred in self.graph.predecessors(self.node):
            if self.cfg.formulation == 'probabilistic':
                pred_uncertain_params[pred] = self.graph.nodes[pred]['parameters_samples']
            elif self.cfg.formulation == 'deterministic':
                pred_uncertain_params[pred] = [{'c': self.graph.nodes[pred]['parameters_best_estimate'], 'w':1.0}]
            else:
                raise ValueError("Invalid formulation.")
            
        return pred_uncertain_params
    
    def prepare_forward_problem_ray(self, inputs, aux):
        """
        Prepares the forward constraints surrogates and decision variables
        """

        # get the inputs from the predecessors of the node
        pred_inputs, pred_auxs = self.get_predecessors_inputs(inputs, aux)
        pred_uncertain_params = self.get_predecessors_uncertain()

        # prepare the forward surrogates
        problem_data = {pred: {p: {} for p in range(len(pred_uncertain_params[pred]))} for pred in self.graph.predecessors(self.node)}
        
        for pred in self.graph.predecessors(self.node):
            for p in range(len(pred_uncertain_params[pred])):
                # load the forward surrogate
                problem_data[pred][p]['equality_constraints'] = self.graph.edges[pred, self.node]["forward_surrogate_serialised"]
                # create a partial function for the optimizer to evaluat                
                if self.cfg.formulation == 'probabilistic':
                    raise NotImplementedError("Method not implemented for probabilistic case")
                    #if self.cfg.solvers.standardised: TODO find a way to handle the case of no classifier training and request for standardisation.
                    #forward_constraints[pred][p] = partial(lambda x, up, inputs: surrogate(jnp.hstack([x.reshape(1,-1), up.reshape(1,-1)])).reshape(-1,1) - inputs.reshape(-1,1), inputs=pred_inputs[pred], up=jnp.array(pred_uncertain_params[pred][p]['c']))
                elif self.cfg.formulation == 'deterministic':
                    if self.cfg.solvers.standardised:   # TODO find a way to handle the case of no classifier training and request for standardisation.
                        pred_input = self.standardise_inputs(jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].copy().reshape(1,-1)]), pred, 'input')
                    else:
                        pred_input = jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].copy().reshape(1,-1)])

                    problem_data[pred][p]['eqc_rhs'] = pred_input.copy().T
                # load the standardised bounds
                decision_bounds = self.graph.nodes[pred]["extendedDS_bounds"].copy()
                if self.cfg.solvers.standardised: decision_bounds = self.standardise_model_decisions(decision_bounds, pred)
                problem_data[pred][p]['bounds'] = decision_bounds.copy()
                # load the forward objective
                problem_data[pred][p]['objective_func'] = self.graph.nodes[pred]["classifier_serialised"]
               
        # return the forward surrogates and decision bounds
        return problem_data
    
    def prepare_forward_problem(self, inputs, aux):
        """
        Prepares the forward constraints surrogates and decision variables
        """

        # get the inputs from the predecessors of the node
        pred_inputs, pred_auxs = self.get_predecessors_inputs(inputs, aux)
        pred_uncertain_params = self.get_predecessors_uncertain()

        # prepare the forward surrogates
        forward_constraints = {pred: {p: None for p in range(len(pred_uncertain_params[pred]))} for pred in self.graph.predecessors(self.node)}
        forward_bounds = {pred: {p: None for p in range(len(pred_uncertain_params[pred]))} for pred in self.graph.predecessors(self.node)}
        forward_objective = {pred: {p: None for p in range(len(pred_uncertain_params[pred]))} for pred in self.graph.predecessors(self.node)}

        
        for pred in self.graph.predecessors(self.node):
            for p in range(len(pred_uncertain_params[pred])):
                # load the forward surrogate
                surrogate = self.graph.edges[pred, self.node]["forward_surrogate"]

                # create a partial function for the optimizer to evaluat                
                if self.cfg.formulation == 'probabilistic':
                    #if self.cfg.solvers.standardised: TODO find a way to handle the case of no classifier training and request for standardisation + update to include auxiliary variables.
                    forward_constraints[pred][p] = partial(lambda x, up, inputs: surrogate(jnp.hstack([x.reshape(1,-1), up.reshape(1,-1)])).reshape(-1,1) - inputs.reshape(-1,1), inputs=jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].copy().reshape(1,-1)]), up=jnp.array(pred_uncertain_params[pred][p]['c']))
                
                elif self.cfg.formulation == 'deterministic':
                    if self.cfg.solvers.standardised:   # TODO find a way to handle the case of no classifier training and request for standardisation.
                        pred_input = self.standardise_inputs(jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].copy().reshape(1,-1)]), pred, 'inputs')
                    else:
                        pred_input = jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].copy().reshape(1,-1)])
                    forward_constraints[pred][p] = partial(lambda x, inputs: surrogate(x.reshape(1,-1)).reshape(-1,1) - inputs.reshape(-1,1), inputs=pred_input.copy())
                else:
                    raise ValueError("Invalid formulation.")
                # load the standardised bounds
                decision_bounds = self.graph.nodes[pred]["extendedDS_bounds"].copy()
                if self.cfg.solvers.standardised: decision_bounds = self.standardise_model_decisions(decision_bounds, pred)
                forward_bounds[pred][p] = decision_bounds.copy()

                # load the forward objective
                classifier = self.graph.nodes[pred]["classifier"]
                forward_objective[pred][p] = lambda x: classifier(x).reshape(-1,1)
               
        # return the forward surrogates and decision bounds
        return forward_objective, forward_constraints, forward_bounds
    
    #def standardise_uncertain_inputs(self, inputs, in_node):

    
    def standardise_inputs(self, inputs, in_node, str_indicator):
        """
        Standardises the inputs
        """
        if str_indicator == 'inputs':
            try:
                mean, std = self.graph.edges[in_node, self.node]['y_scalar'].mean, self.graph.edges[in_node,self.node]['y_scalar'].std
                return (inputs - mean) / std
            except:

                return inputs
        elif str_indicator == 'aux':
            try:
                mean, std = self.graph.edges[in_node, self.node]['aux_scalar'].mean, self.graph.edges[in_node,self.node]['aux_scalar'].std
                return (inputs - mean) / std
            except:
                return inputs
        else:
            raise ValueError("Invalid indicator.")
    
    def standardise_model_decisions(self, decisions, in_node):
        """
        Standardises the decisions
        """
        try:
            mean, std = self.graph.nodes[in_node]['classifier_x_scalar'].mean, self.graph.nodes[in_node]['classifier_x_scalar'].std
            return [(decision - mean) / std for decision in decisions]
        except:
            try:
                mean, std = self.graph.nodes[in_node]['x_scalar'].mean, self.graph.nodes[in_node]['x_scalar'].std
                return [(decision - mean) / std for decision in decisions]
            except:
                return decisions
        
        
def assess_feasibility(feasibility, input):
    """
    Assesses the feasibility of the input
    """
    if feasibility == 'positive':
        return input >= 0
    elif feasibility == 'negative':
        return input <= 0
    else:
        raise ValueError("Invalid notion of feasibility.")
    

""" ---- JaxOpt solver evaluation methods (written as pure functions not classes) --- """

# TODO adapt to the classifier masking to generalise to more than two in-neighbours.

def shaping_function(x, cfg):
    """
    Shaping function
    """
    if cfg.samplers.notion_of_feasibility == 'positive':
        return -x
    elif cfg.samplers.notion_of_feasibility == 'negative':
        return x

def get_successor_inputs(graph, node, outputs):
    """
    Gets the inputs from the predecessors
    """
    succ_inputs = {}
    for succ in graph.successors(node):
        output_indices = graph.edges[node, succ]['edge_fn']
        if outputs.ndim < 2: outputs = outputs.reshape(-1, 1)
        if outputs.ndim < 3: outputs = jnp.expand_dims(outputs, axis=0)
        succ_inputs[succ]= output_indices(outputs).reshape(outputs.shape[1],-1)
    return succ_inputs


def construct_solver(objective_func, bounds, tol):
    bounds = bounds
    objective_func = objective_func
    bounds = bounds
    solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=objective_func, bounds_=(bounds[0], bounds[1]), tol=tol)
    return solver   

def initial_guess(cfg, bounds):
    n_d = len(bounds[0])
    return generate_initial_guess(cfg.n_starts, n_d, bounds)

def solve(solver, initial_guesses):
    obj_r, e  = [], []

    for solve, init in zip(solver, initial_guesses):
        objective, error = solve(init)
        obj_r.append(objective)
        e.append(error)
    
    return {'objective': jnp.array(obj_r), 'error': jnp.array(e)}
    

def load_solver(objective_func, bounds):
    """
    Loads the solver
    """
    return construct_solver(objective_func, bounds)


def prepare_backward_problem(outputs, graph, node, cfg):
    """
    Prepares the forward constraints surrogates and decision variables
    - ouptuts from a nodes unit functions are inputs to the next unit

    """
    backward_bounds = {succ: None for succ in graph.successors(node)}
    backward_objective = {succ: None for succ in graph.successors(node)}

    # get the outputs from the successors of the node
    succ_inputs = get_successor_inputs(graph, node, outputs)

    for succ in graph.successors(node):

        n_d  = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices'].copy()]))
        aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices'].copy()]))
        
        # standardisation of outputs if required
        if cfg.solvers.standardised: succ_inputs[succ] = succ_inputs[succ].at[:].set(standardise_inputs(graph, succ_inputs[succ].copy(), succ, jnp.hstack([input_indices, aux_indices])))
        
        # load the standardised bounds
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        ndim = graph.nodes[succ]['n_design_args'] + graph.nodes[succ]['n_input_args'] + graph.graph['n_aux_args']
        
        # get the decision bounds
        if cfg.solvers.standardised: decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        
        decision_bounds = [jnp.delete(bound, np.hstack([input_indices,aux_indices])) for bound in decision_bounds]
        backward_bounds[succ] = [decision_bounds.copy() for i in range(succ_inputs[succ].shape[0])]

        # load the forward objective
        classifier = graph.nodes[succ]["classifier"]
        wrapper_classifier = mask_classifier(classifier, n_d, ndim, input_indices, aux_indices)
        backward_objective[succ] = [jit(partial(lambda x,y: wrapper_classifier(x,y).squeeze(), y=succ_inputs[succ][i].reshape(1,-1))) for i in range(succ_inputs[succ].shape[0])]

    # return the forward surrogates and decision bounds
    return backward_objective, backward_bounds


def evaluate(outputs, aux, graph, node, cfg):
    """
    Evaluates the constraints
    """
    evaluate_method = solve
    objective, bounds = prepare_backward_problem(outputs, graph, node, cfg)
    succ_fn_evaluations = {}
    # iterate over successors and evaluate the constraints
    for succ in graph.successors(node):
        backward_solver = [construct_solver(objective[succ][i], bounds[succ][i], tol=cfg.solvers.backward_coupling.jax_opt_options.error_tol) for i in range(outputs.shape[0])] # uncertainty realizations evaluated serially
        initial_guesses = [initial_guess(cfg.solvers.backward_coupling, bounds[succ][i]) for i in range(outputs.shape[0])]
        succ_fn_evaluations[succ] = evaluate_method(backward_solver, initial_guesses)

    # reshape the evaluations
    fn_evaluations = [succ_fn_evaluations[succ]['objective'].reshape(-1,1) for succ in graph.successors(node)] # TODO make sure this is compatible with format returned by solver

    return shaping_function(jnp.hstack(fn_evaluations), cfg)

def construct_input(x, y, fix_ind, aux_ind, ndim):
    # Initialize input_ with zeros or any placeholder value
    input_ = jnp.zeros(ndim)
    
    # Create a mask for positions not in fix_ind and aux_ind
    total_indices = jnp.arange(ndim)
    opt_ind = jnp.delete(total_indices, np.concatenate([fix_ind, aux_ind]))
    
    # Assign values from x to input_ at positions not in fix_ind and aux_ind
    input_ = input_.at[opt_ind].set(x)
    
    # Assign values from y to input_ at positions in fix_ind and aux_ind
    input_ = input_.at[fix_ind].set(y[0,:len(fix_ind)])
    input_ = input_.at[aux_ind].set(y[0,len(fix_ind):])
    
    return input_


def mask_classifier(classifier, nd, ndim, fix_ind, aux_ind):
    """
    Masks the classifier
    - y corresponds to those indices that are fixed
    - x corresponds to those indices that are optimised
    # NOTE this is not general to nodes with more than two in-neighbours
    """

    def masked_classifier(x, y):
        input_ = construct_input(x, y, fix_ind, aux_ind, ndim)
        return classifier(input_.reshape(1,-1)).squeeze()
    
    return jit(masked_classifier)

def standardise_inputs(graph, succ_inputs, out_node, input_indices):
    """
    Standardises the inputs
    """
    standardiser = graph.nodes[out_node]['classifier_x_scalar']
    if standardiser is None: return succ_inputs
    else:
        mean, std = standardiser.mean, standardiser.std
        return (succ_inputs - mean[input_indices].reshape(1,-1)) / std[input_indices].reshape(1,-1)

def standardise_model_decisions(graph, decisions, out_node):
    """
    Standardises the decisions
    """
    standardiser = graph.nodes[out_node]['classifier_x_scalar']
    if standardiser is None: return decisions
    else:
        mean, std = standardiser.mean, standardiser.std
        return [(decision - mean[:]) / std[:] for decision in decisions]


def jax_pmap_evaluator(outputs, aux, cfg, graph, node):
    """
    p-map constraint evaluation call - called by backward_surrogate_pmap_batch_evaluator
    """
    constraint_evaluator = partial(evaluate, graph=graph, node=node, cfg=cfg)

    return constraint_evaluator(outputs, aux)



def backward_surrogate_pmap_batch_evaluator(outputs, aux, cfg, graph, node):
    """
    Evaluates the constraints on a batch using jax-pmap - called by the backward_constraint_evaluator
    """
    feasibility_call = partial(jax_pmap_evaluator, cfg=cfg.copy(), graph=graph.copy(), node=node)
    
    return pmap(feasibility_call, in_axes=(0,0), out_axes=0, devices=[device for i, device in enumerate(devices('cpu')) if i<outputs.shape[0]])(outputs, aux)  #, axis_name='i'
   
 

def backward_constraint_evaluator(outputs, aux, cfg, graph, node, pool):
    """
    Evaluates the constraints using jax-pmap - this is what should be called

    Syntax: 
        call: method_(outputs, cfg, graph, node, pool)

    """
    max_devices = cfg.max_devices
    batch_sizes, remainder = determine_batches(outputs.shape[0], max_devices)
    # get batches of outputs
    output_batches = create_batches(batch_sizes, outputs)
    aux_batches = create_batches(batch_sizes, jnp.repeat(jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1))
    # evaluate the constraints
    results = []
    for i, (output_batch, aux_batch) in enumerate(zip(output_batches, aux_batches)):
        results.append(backward_surrogate_pmap_batch_evaluator(output_batch, aux_batch, cfg, graph, node))
    # concatenate the results
    return jnp.vstack(results)

    
def lower_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    # compute the lower bound of the likelihood
    # constraint_evals: the constraint evaluations
    # samples: the number of samples we used
    # confidence: the desired confidence level

    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    # compute the average constraint evaluation
    F_vioSA = jnp.mean(constraint_evals)

    # compute alpha and beta for the beta distribution
    alpha = samples + 1 - samples * F_vioSA
    b_ta = samples * F_vioSA + 1e-8

    # compute the lower bound of the likelihood as the inverse of the CDF of the beta distribution
    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(conf)

    return 1 - F_LB


def upper_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    # compute the lower bound of the likelihood
    # constraint_evals: the constraint evaluations
    # samples: the number of samples we used
    # confidence: the desired confidence level

    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    # compute the average constraint evaluation
    F_vioSA = jnp.mean(constraint_evals)

    # compute alpha and beta for the beta distribution
    alpha = samples - samples * F_vioSA
    b_ta = samples * F_vioSA + 1

    # compute the upper bound of the likelihood as the inverse of the CDF of the beta distribution
    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(1 - conf)

    return 1 - F_LB

if __name__ == "__main__":
    
    from constructor import constructor_test
    constructor_test()

    