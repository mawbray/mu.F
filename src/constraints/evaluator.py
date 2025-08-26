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
        self.cfg = cfg
        self.graph = graph
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
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['constraints_vmap'].copy())
        else:
            return list(self.graph.nodes[self.node]['constraints'].copy())
    
    def vmap_evaluation(self):
        """
        Vectorizes the the constraints and then loads them back onto the graph
        """
        # get constraints from the graph
        constraints = self.graph.nodes[self.node]['constraints'].copy()
        # vectorize each constraint
        cons = [jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model), in_axes=(0), out_axes=0)), in_axes=(1), out_axes=1)) for constraint in constraints]
        # load the vectorized constraints back onto the graph
        self.graph.nodes[self.node]['constraints_vmap'] = cons

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
            self.evaluation_method = self.serial_evaluation 
        if self.pool == 'ray':
            self.evaluation_method = self.ray_evaluation
        
    def __call__(self, inputs, aux):
        return self.evaluate(inputs, aux)
    
    def serial_evaluation(self, solver, max_devices, solver_processing):

        # determine the batch size
        workers, remainder = determine_batches(len(solver), 1)
        # split the problems
        solver_batches = create_batches(workers, solver)

        # parallelise the batch
        result_dict = {}
        evals = 0
        for i, solve in enumerate(solver_batches): 
            results = [sol(d['id'], d['data'], d['data']['cfg']) for sol, d in  solve] # set off and then synchronize before moving on
            for j, result in enumerate(results):
                result_dict[evals + j] = solver_processing.solve_digest(*result)['objective']
            evals += j+1

        del solver_batches, results
    
        return jnp.concatenate([jnp.array([value]).reshape(1,-1) for _, value in result_dict.items()], axis=0)
    
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
        return self.wrapper(inputs, aux)
    
    
    def wrapper(self, inputs, aux):
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """

        solver_inputs = []
        # get solvers for each problem
        for i in range(inputs.shape[0]):
            solver_inputs.append(self.evaluate_parallel(i, inputs[i,:].reshape(1,-1), aux[i,:].reshape(1,-1)))        

        if len(list(si.keys() for si in [list(solver_inputs[0].values())[0]])) > 1:
            raise NotImplementedError("Case of uncertainty in forward pass not yet implemented/optimised for parallel evaluation.")
        else:
            results = {prec: [] for prec in self.graph.predecessors(self.node)}
            # run solvers in parallel
            for prec in self.graph.predecessors(self.node):
                solver_reshape = []
                # NOTE currently this iterates over uncertainty realisations (although not actually implemented in the code following) - think about expectations here.)
                for p in range(len(solver_inputs[0][prec])):
                    for s_i in solver_inputs:
                        solver_reshape.append((s_i[prec][p].solver, s_i[prec][p].problem_data))
                # evaluate inputs for in parallel for each evaluation of uncertainty
                results[prec].append(self.evaluation_method(solver_reshape, self.cfg.max_devices, s_i[prec][p]))


            return jnp.concatenate([jnp.array(v).reshape(-1,1) for v in results.values()], axis=-1)
            
    
    def evaluate_parallel(self, i, inputs, auxs):

        """
        Evaluates the constraints
        """
        problem_data = self.prepare_forward_problem(inputs, auxs)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_input_i = {pred: {} for pred in self.graph.predecessors(self.node)}

        solved_successful = 0
        problems = sum([len(problem_data[pred]) for pred in self.graph.predecessors(self.node)])

        # iterate over predecessors and evaluate the constraints
        for pred in self.graph.predecessors(self.node):
            for p in range(len(problem_data[pred])): 
                forward_solver = solver_object.from_method(self.cfg.solvers.forward_coupling, solver_object.solver_type, problem_data[pred][p]['objective_func'], problem_data[pred][p]['bounds'], problem_data[pred][p]['constraints'])
                initial_guess = forward_solver.initial_guess()
                forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
                forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data[pred][p]['eq_rhs']
                forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data[pred][p]['eq_lhs']
                forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
                forward_solver.solver.problem_data['data']['uncertain_params'] = None
                forward_solver.solver.problem_data['id'] = i
                pred_fn_input_i[pred][p] = forward_solver.solver

        return pred_fn_input_i

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
    
    def prepare_forward_problem(self, inputs, aux):
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
                if self.cfg.formulation == 'probabilistic':
                    raise NotImplementedError("Method not implemented for probabilistic case")
                    #if self.cfg.solvers.standardised: TODO find a way to handle the case of no classifier training and request for standardisation.
                elif self.cfg.formulation == 'deterministic':
                    if self.cfg.solvers.standardised:   # TODO find a way to handle the case of no classifier training and request for standardisation.
                        pred_input = self.standardise_inputs(jnp.hstack([pred_inputs[pred].copy().reshape(1,-1), pred_auxs[pred].reshape(1,-1)]), pred, 'inputs')
                    else:
                        pred_input = jnp.hstack([pred_inputs[pred].reshape(1,-1), pred_auxs[pred].reshape(1,-1)])

                    problem_data[pred][p]['eq_rhs'] = pred_input.T
                    problem_data[pred][p]['eq_lhs'] = pred_input.T

                n_d_j = self.graph.nodes[pred]['n_design_args'] + self.graph.nodes[pred]['n_input_args'] + self.graph.graph['n_aux_args']    
                
                # load the forward surrogate
                problem_data[pred][p]['constraints'] = {0: {'params':self.graph.edges[pred, self.node]["forward_surrogate_serialised"], 
                                                        'args': [i for i in range(n_d_j)], 
                                                        'model_class': 'regression', 'model_surrogate': 'forward_evaluation_surrogate', 
                                                        'model_type': self.cfg.surrogate.regressor_selection, 
                                                        'g_fn': partial(lambda x, fn, v : fn(x.reshape(1,-1)[:,v]).reshape(-1,1), v = [i for i in range(n_d_j)])}}
                
                # load the forward objective
                prev_node_backoff = self.graph.nodes[pred]['constraint_backoff']
                problem_data[pred][p]['objective_func'] = {'f0': {'params': self.graph.nodes[pred]["classifier_serialised"], 
                                                                  'args': [i for i in range(n_d_j)],
                                                                  'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                                  'model_type': self.cfg.surrogate.classifier_selection},
                                                                  'obj_fn': partial(lambda x, f1, b, t: f1(x.reshape(1,-1)[:,t]).reshape(-1,1) + b, b=prev_node_backoff, t=[i for i in range(n_d_j)])}
                
                # load the standardised bounds
                decision_bounds = self.graph.nodes[pred]["extendedDS_bounds"]
                if self.cfg.solvers.standardised: decision_bounds = self.standardise_model_decisions(decision_bounds, pred)
                # load the constraints from l \in N_j^out
                k_index=1
                last_index = n_d_j
                if self.cfg.solvers.forward_general_constraints:
                    for l_post in self.graph.successors(pred):
                        if l_post < self.node: #  if l comes before the current node in the graph then lets add constraints
                            n_d_l = self.graph.nodes[l_post]['n_design_args'] + self.graph.nodes[l_post]['n_input_args'] + self.graph.graph['n_aux_args']
                            n_design_l, n_input_indices_l, n_auxiliary = self.graph.nodes[l_post]['n_design_args'], self.graph.edges[pred, l_post]['input_indices'], self.graph.graph['n_aux_args']
                            problem_data[pred][p]['constraints'][k_index] = {'params': self.graph.edges[pred, l_post]["forward_surrogate_serialised"], 
                                                                            'args': [i for i in range(last_index + n_d_l)],
                                                                            'model_class': 'regression', 'model_surrogate': 'forward_evaluation_surrogate', 
                                                                            'model_type': self.cfg.surrogate.regressor_selection,
                                                                            'g_fn': partial(lambda x, fn, v, l: fn(x.reshape(1,-1)[:, v]).reshape(-1,1) - x.reshape(-1,1)[l,:], 
                                                                                            v=[i for i in range(n_d_j)], l=[i +last_index + n_design_l for i in n_input_indices_l])}
                            k_index += 1
                            # load the forward surrogate inequality constraint
                            problem_data[pred][p]['constraints'][k_index] = {'params': self.graph.nodes[l_post]["classifier_serialised"], 
                                                                            'args': [ i for i in range(last_index+n_d_l)],
                                                                            'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                                            'model_type': self.cfg.surrogate.classifier_selection,
                                                                            'g_fn': partial(lambda x, fn, v : fn(x.reshape(1,-1)[:,v]).reshape(-1,1), v = [last_index + i for i in range(n_d_l)])}
                            k_index += 1
                            last_index += n_d_l 
                            problem_data[pred][p]['eq_rhs'] = jnp.vstack([problem_data[pred][p]['eq_rhs'], jnp.zeros(len(n_input_indices_l)+1,).reshape(-1,1)])
                            problem_data[pred][p]['eq_lhs'] = jnp.vstack([problem_data[pred][p]['eq_lhs'], jnp.zeros(len(n_input_indices_l),).reshape(-1,1), -jnp.ones(1,).reshape(-1,1)*jnp.inf])
                            # add (un)standardised bounds 
                            db = self.graph.nodes[l_post]["extendedDS_bounds"].copy()
                            if self.cfg.solvers.standardised: db = self.standardise_model_decisions(db, l_post) 
                            decision_bounds = [jnp.hstack([decision_bounds[0], db[0]]), jnp.hstack([decision_bounds[1], db[1]])]

                            # issue with LBG atm
                    
                problem_data[pred][p]['bounds'] = decision_bounds
                
        # return the forward surrogates and decision bounds
        return problem_data
    
    
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
            mean, std = self.graph.nodes[in_node]['classifier_x_scalar'].mean, self.graph.nodes[in_node]['classifier_x_scalar'].std  #NOTE this indexing needs addressing in probabilistic case
            return [(decision - mean) / std for decision in decisions]
        except:
            try:
                mean, std = self.graph.nodes[in_node]['x_scalar'].mean, self.graph.nodes[in_node]['x_scalar'].std
                return [(decision - mean) / std for decision in decisions]
            except:
                return decisions
        


class backward_constraint_evaluator_general(forward_constraint_evaluator):
    """
    A backward surrogate constraint
    - solved using casadi interface with jax and IPOPT
    - parallelism is provided by multiprocessing pool
        : may be extended to jax-pmap in the future if someone develops a nice nlp solver in jax

    Syntax: 
        initialise: class_instance(cfg, graph, node, pool)
        call: class_instance.evaluate(inputs)

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg, graph, node, pool=None):
        super().__init__(cfg, graph, node, pool)
        # pool settings
        self.pool = pool
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool == 'jax-pmap':
            raise NotImplementedError("jax-pmap is not supported for this constraint type at the moment.")
        if self.pool == 'mp-ms':
            self.evaluation_method = self.serial_evaluation 
        if self.pool == 'ray':
            self.evaluation_method = self.ray_evaluation
    
    def __call__(self, outputs, aux):
        return self.evaluate(outputs)

    def evaluate(self, outputs):
        """
        Evaluates the constraints by iterating sequentially over the design and the uncertain params
        inputs: samples in the extended design space
        outputs: the constraint evaluations
        """
        return self.wrapper(outputs)
    
    
    def wrapper(self, outputs):
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """

        solver_inputs = []
        # get solvers for each problem
        for i in range(outputs.shape[0]):
            solver_inputs.append(self.evaluate_parallel(i, outputs[i,:].reshape(1,-1)))        

        if len(list(solver_inputs[0].values())) > 1:
            raise NotImplementedError("Case of uncertainty in forward pass not yet implemented/optimised for parallel evaluation.")
        else:

            results = {succ: [] for succ in self.graph.successors(self.node)}
            # run solvers in parallel
            for succ in self.graph.successors(self.node):
                solver_reshape = []
                # NOTE currently this iterates over uncertainty realisations (although not actually implemented in the code following) - think about expectations here.)
                for p in range(len(solver_inputs[0][succ])):
                    for s_i in solver_inputs:
                        solver_reshape.append((s_i[succ][p].solver, s_i[succ][p].problem_data))
                # evaluate inputs for in parallel for each evaluation of uncertainty
                results[succ].append(self.evaluation_method(solver_reshape, self.cfg.max_devices, s_i[succ][p]))


            return jnp.concatenate([jnp.array(v).reshape(-1,1) for v in results.values()], axis=-1)
            
    
    def prepare_forward_problem(self, outputs):
        """
        Prepares the constraints surrogates and decision variables
        """

        # get the outputs from the successors of the node
        graph, node, cfg = self.graph, self.node, self.cfg

        backward_bounds = {succ: None for succ in graph.successors(node)}
        backward_objective = {succ: None for succ in graph.successors(node)}


        succ_inputs = get_successor_inputs(graph, node, outputs)
        # prepare the forward surrogates
        problem_data = {succ: {p: {} for p in range(succ_inputs[succ].shape[1])} for succ in self.graph.successors(self.node)}


        for succ in self.graph.successors(self.node):
            assert succ_inputs[succ].shape[1]  == 1, f"Problem data shape mismatch: {succ_inputs[succ].shape[1]} != {1}"

        p=0 
        for succ in self.graph.successors(self.node):
            
            if self.cfg.formulation == 'probabilistic':
                raise NotImplementedError("Method not implemented for probabilistic case")
                #if self.cfg.solvers.standardised: TODO find a way to handle the case of no classifier training and request for standardisation.
            elif self.cfg.formulation == 'deterministic':
                
                n_d  = graph.nodes[succ]['n_design_args']
                input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
                edge_input_specific_indices= np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
                aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]))
                
                # standardisation of outputs if required
                if cfg.solvers.standardised: succ_inputs[succ] = succ_inputs[succ].at[:].set(standardise_inputs(graph, succ_inputs[succ], succ, jnp.hstack([input_indices, aux_indices]).astype(int)))
                
                # load the standardised bounds
                decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
                ndim = graph.nodes[succ]['n_design_args'] + graph.nodes[succ]['n_input_args'] + graph.graph['n_aux_args']
                
                # get the decision bounds
                if cfg.solvers.standardised: decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
                decision_bounds = [jnp.delete(bound, np.hstack([edge_input_specific_indices,aux_indices]).astype(int), axis=1) for bound in decision_bounds]

                # --- equality constraints reduced into objective using output data  --- #
                problem_data[succ][p]['eq_rhs'] = jnp.empty((0,1))
                problem_data[succ][p]['eq_lhs'] = jnp.empty((0,1))
                n_d_k = self.graph.nodes[succ]['n_design_args'] + sum([self.graph.edges[n,succ]['n_input_args'] for n in self.graph.predecessors(succ) if n!=self.node]) + self.graph.graph['n_aux_args']    
                
                # load the objective
                problem_data[succ][p]['objective_func'] = {'f0': {'params': self.graph.nodes[succ]["classifier_serialised"], 
                                                                  'args': [i for i in range(n_d_k)],
                                                                  'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                                  'model_type': self.cfg.surrogate.classifier_selection},
                                                                  'obj_fn': partial(lambda x, f1, y: mask_classifier(f1, n_d, ndim, input_indices, aux_indices)(x.reshape(1,-1)[:,:n_d_k],y).reshape(-1,1), y=succ_inputs[succ].reshape(1,-1))}
                
                assert len(jnp.delete(jnp.arange(ndim), np.concatenate([input_indices, aux_indices]).astype(int))) == n_d_k, 'shape mismatch in the masking and the decision variables'

                # load the standardised bounds
                decision_bounds = self.graph.nodes[succ]["extendedDS_bounds"]
                if self.cfg.solvers.standardised: decision_bounds = self.standardise_model_decisions(decision_bounds, succ)
                
                problem_data[succ][p]['constraints'] = {}
                k_index = 0
                last_index = n_d_k
                for m_prec in self.graph.predecessors(succ):
                    if m_prec > self.node: #  if lm comes after the current node in the graph then lets add constraints
                        n_d_m = self.graph.nodes[m_prec]['n_design_args'] + self.graph.nodes[m_prec]['n_input_args'] + self.graph.graph['n_aux_args']
                        n_design_m, n_input_indices_m, n_auxiliary = self.graph.nodes[m_prec]['n_design_args'], self.graph.edges[m_prec, succ]['input_indices'], self.graph.graph['n_aux_args']
                        problem_data[succ][p]['constraints'][k_index] = {'params': self.graph.edges[m_prec, succ]["forward_surrogate_serialised"], 
                                                                        'args': [i for i in range(last_index + n_d_m)],
                                                                        'model_class': 'regression', 'model_surrogate': 'forward_evaluation_surrogate', 
                                                                        'model_type': self.cfg.surrogate.regressor_selection,
                                                                        'g_fn': partial(lambda x, fn, v, l: fn(x.reshape(1,-1)[:, v]).reshape(-1,1) - x.reshape(-1,1)[l,:], 
                                                                                        v=[last_index+ i for i in range(n_d_m)], l=[i +n_d for i in n_input_indices_m])}
                        k_index += 1
                        # load the forward surrogate inequality constraint
                        problem_data[succ][p]['constraints'][k_index] = {'params': self.graph.nodes[m_prec]["classifier_serialised"], 
                                                                        'args': [ i for i in range(last_index+n_d_m)],
                                                                        'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                                        'model_type': self.cfg.surrogate.classifier_selection,
                                                                        'g_fn': partial(lambda x, fn, v : fn(x.reshape(1,-1)[:,v]).reshape(-1,1), v = [last_index+ i for i in range(n_d_m)])}
                        k_index += 1
                        last_index += n_d_m     
                        problem_data[succ][p]['eq_rhs'] = jnp.vstack([problem_data[succ][p]['eq_rhs'], jnp.zeros(len(n_input_indices_m)+1,).reshape(-1,1)])
                        problem_data[succ][p]['eq_lhs'] = jnp.vstack([problem_data[succ][p]['eq_lhs'], jnp.zeros(len(n_input_indices_m),).reshape(-1,1), -jnp.inf*jnp.ones((1,)).reshape(-1,1)])
                        
                        # add (un)standardised bounds 
                        db = self.graph.nodes[m_prec]["extendedDS_bounds"].copy()
                        if self.cfg.solvers.standardised: db = self.standardise_model_decisions(db, m_prec) 
                        decision_bounds = [jnp.hstack([decision_bounds[0], db[0]]), jnp.hstack([decision_bounds[1], db[1]])]

                problem_data[succ][p]['bounds'] = decision_bounds
                
        # return the forward surrogates and decision bounds
        return problem_data
    
    def evaluate_parallel(self, i, outputs):

        """
        Evaluates the constraints
        """
        problem_data = self.prepare_forward_problem(outputs)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        succ_fn_input_i = {succ: {} for succ in self.graph.successors(self.node)}

        solved_successful = 0
        problems = sum([len(problem_data[pred]) for pred in self.graph.successors(self.node)])

        # iterate over successors and evaluate the constraints
        for succ in self.graph.successors(self.node):
            for p in range(len(problem_data[succ])): 
                forward_solver = solver_object.from_method(self.cfg.solvers.forward_coupling, solver_object.solver_type, problem_data[succ][p]['objective_func'], problem_data[succ][p]['bounds'], problem_data[succ][p]['constraints'])
                initial_guess = forward_solver.initial_guess()
                forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
                forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data[succ][p]['eq_rhs']
                forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data[succ][p]['eq_lhs']
                forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
                forward_solver.solver.problem_data['data']['uncertain_params'] = None
                forward_solver.solver.problem_data['id'] = i
                succ_fn_input_i[succ][p] = forward_solver.solver

        return succ_fn_input_i


class q_learning_evaluator(backward_constraint_evaluator_general):
    """
    A Q-learning based evaluator for backward constraints
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node, pool)

    def prepare_forward_problem(self, outputs):
        problem_data = super().prepare_forward_problem(outputs)

        rewards = self.graph.nodes[self.node]['reward_function'](outputs)

        # My thoughts are to add the q function evaluation to the logic here
        # Then to move the objective function from problem data into the constraints. 
        succ = self.graph.successors(self.node)[0]  # Type = list of nodes, we only have one succ.
        problem_data[succ]['constraints'][0] = problem_data[succ]['objective_func']['f0']
        problem_data[succ]['constraints'][0]['g_fn'] = problem_data[succ]['objective_func']['obj_fn']

        # Get the necessary inputs for the successor
        graph, node, cfg = self.graph, self.node, self.cfg
        succ_input = get_successor_inputs(graph, node, outputs)[succ]
        n_d = self.graph.nodes[succ]['n_design_args']
        ndim = self.graph.nodes[succ]['ndim']
        input_indices = self.graph.nodes[succ]['input_indices']
        aux_indices = self.graph.nodes[succ]['aux_indices']
        n_d_k = self.graph.nodes[succ]['n_design_args'] + sum([self.graph.edges[n,succ]['n_input_args'] for n in self.graph.predecessors(succ) if n!=self.node]) + self.graph.graph['n_aux_args']    


        # Then we need to redefine the objective function for our q-learning target.
        problem_data[succ]['objective_func'] = {'f0': {
            'params': self.graph.nodes[succ]["q_function_serialised"], # TODO <- Need to change how the name is saved for the q function surrogate. 
            'args': [i for i in range(n_d_k)],
            'model_class': 'regression', 'model_surrogate': 'q_func_surrogate', 
            'model_type': self.cfg.surrogate.q_function_selection},
            'obj_fn': partial(lambda x, f1, y: mask_classifier(f1, n_d, ndim, input_indices, aux_indices)(x.reshape(1,-1)[:,:n_d_k],y).reshape(-1,1), y=succ_input.reshape(1,-1))}

        return problem_data

class forward_constraint_decentralised_evaluator(forward_constraint_evaluator):
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
        super().__init__(cfg, graph, node, pool)
        # pool settings
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
        
        if self.pool == 'ray':
            for i, solve in enumerate(solver_batches):
                results = ray.get([sol.remote(d['id'], d['data'], d['data']['cfg']) for sol, d in  solve]) # set off and then synchronize before moving on
                for j, result in enumerate(results):
                    result_dict[evals + j] = solver_processing.solve_digest(*result)['objective']
                evals += j+1


        del solver_batches, results

        return -jnp.concatenate([jnp.array([value]).reshape(1,-1) for _, value in result_dict.items()], axis=0)

    
    def ray_wrapper(self, inputs, aux):
        """ first prepare the problem set up, 
        then evaluate the constraints in parallel using ray.
        """

        solver_inputs = []
        for i in range(inputs.shape[0]):
            solver_inputs.append(self.evaluate_parallel(i, inputs[i,:].reshape(1,-1)))

        if len(list(solver_inputs[0])) > 1:
            raise NotImplementedError("Case of uncertainty in forward pass not yet implemented/optimised for parallel evaluation.")
        else:
            results = []
            solver_reshape = []
            for p in range(len(solver_inputs[0])):
                for s_i in solver_inputs:
                    solver_reshape.append((s_i[p].solver, s_i[p].problem_data))
            results.append(self.ray_evaluation(solver_reshape, self.cfg.max_devices, s_i[p]))


            return jnp.concatenate(results, axis=-1)


    def evaluate_parallel(self, i, inputs, aux):
        """
        Evaluates the constraints
        """
        problem_data = self.prepare_forward_problem(inputs, aux)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_input_i = {0: {0: {}}}
        # iterate over predecessors and evaluate the constraints
        for pred in range(1):
            for p in range(1): 
                forward_solver = solver_object.from_method(self.cfg.solvers.forward_coupling, solver_object.solver_type, problem_data[pred][p]['objective_func'], problem_data[pred][p]['bounds'], problem_data[pred][p]['constraints'])
                initial_guess = forward_solver.initial_guess()
                forward_solver.solver.problem_data['data']['initial_guess'] = initial_guess
                forward_solver.solver.problem_data['data']['eq_rhs'] = problem_data[pred][p]['eq_rhs']
                forward_solver.solver.problem_data['data']['eq_lhs'] = problem_data[pred][p]['eq_lhs']
                forward_solver.solver.problem_data['data']['cfg'] = dict(self.cfg)
                forward_solver.solver.problem_data['data']['uncertain_params'] = None
                forward_solver.solver.problem_data['id'] = i
                pred_fn_input_i[pred][p] = forward_solver.solver

        return {0: {0: forward_solver.solver}}
    

    def prepare_forward_problem(self, inputs, aux):        
        """
        Prepares the forward constraints surrogates and decision variables
        """

        # get the inputs from the predecessors of the node
        pred_inputs, pred_auxs = self.get_predecessors_inputs(inputs, aux)
        pred_uncertain_params = self.get_predecessors_uncertain()
        problem_data = {0: {0: {} }}
                       
        # prepare the forward surrogates
        x = [self.graph.nodes[pred]['n_design_args'] + self.graph.nodes[pred]['n_input_args'] + self.graph.graph['n_aux_args'] for pred in self.graph.predecessors(self.node)]
        t = np.cumsum(x)
        n_d_j = sum(x)
        prev_node_backoff = [self.graph.nodes[pred]['constraint_backoff'] for pred in self.graph.predecessors(self.node)]

        # load the predecessors surrogates
        problem_data[0][0]['constraints'] = {i: {'params':self.graph.nodes[pred]["classifier_serialised"], 
                                                'args': [i for i in range(n_d_j)], 
                                                'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                'model_type': self.cfg.surrogate.classifier_selection, 
                                                'g_fn': partial(lambda x, fn, v, b : fn(x.reshape(1,-1)[:,v]).reshape(-1,1) + b, v = [t[i] + k for k in range(x[i])], b = prev_node_backoff[i])} for i, pred in enumerate(self.graph.predecessors(self.node))}
        
        # load the lhs and rhs of the equality constraints
        n_p  = len([p for p in self.graph.predecessors(self.node)])
        problem_data[0][0]['eq_lhs'] = -jnp.ones(n_p,).reshape(-1,1)*jnp.inf
        problem_data[0][0]['eq_rhs'] = -jnp.zeros(n_p,).reshape(-1,1)

        # load the forward objective
        problem_data[0][0]['objective_func'] = {'f0': {'params': self.graph.nodes[self.node]["classifier_serialised"], 
                                                            'args': [i for i in range(n_d_j)],
                                                            'model_class': 'classification', 'model_surrogate': 'live_set_surrogate', 
                                                            'model_type': self.cfg.surrogate.classifier_selection},
                                                            'obj_fn': partial(lambda x, f1, b, t: f1(x.reshape(1,-1)[:,t]).reshape(-1,1) + b, b=prev_node_backoff, t=[i for i in range(n_d_j)])}
        
        node_backoff = self.graph.nodes[self.node]['constraint_backoff']
        problem_data[0][0]['objective_func']['obj_fn']= partial(lambda x, f1, f2, v, b: - f1(jnp.hstack([v, f2(x.reshape(1,-1)).reshape(1,-1)])).reshape(-1,1) - b, v=jnp.hstack(inputs).reshape(1,-1), b=node_backoff)

        for i, pred in enumerate(self.graph.predecessors(self.node)):
            problem_data[0][0]['objective_func'][f'f{i+1}'] = {'params': self.graph.edges[pred,self.node]["forward_surrogate_serialised"], 
                                                                'args': [t[i] + k for k in range(x[i])],
                                                                'model_class': 'regression', 
                                                                'model_surrogate': 'forward_evaluation_surrogate', 
                                                                'model_type': self.cfg.surrogate.regressor_selection}
        
        # load the standardised bounds
        decision_bounds = [[], []]
        for i, pred in enumerate(self.graph.predecessors(self.node)):
            decisions = self.graph.nodes[pred]["extendedDS_bounds"].copy()
            if self.cfg.solvers.standardised:
                decisions = self.standardise_model_decisions(decisions, pred)
            decision_bounds[0].extend(decisions[0])
            decision_bounds[1].extend(decisions[1])

        decision_bounds = [jnp.array(decision_bounds[0]), jnp.array(decision_bounds[1])]

            
        problem_data[0][0]['bounds'] = decision_bounds
                
        # return the forward surrogates and decision bounds
        return problem_data



class forward_root_constraint_decentralised_evaluator(coupling_surrogate_constraint_base):
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
        
    def __call__(self, inputs, aux):
        return self.evaluate(inputs, aux)


    def evaluate(self, design, aux):
        """
        Evaluates the constraints by iterating sequentially over the design and the uncertain params
        inputs: samples in the extended design space
        outputs: the constraint evaluations
        """
        return self.evaluate_vmap(design, aux)

            
    def evaluate_vmap(self, decisions, aux):
        """
        Evaluates the constraints
        """
        constraints, inputs = self.prepare_forward_problem(jnp.hstack([decisions, aux]))
        g_vals = constraints(inputs)

        del constraints, inputs

        return self.shaping_function(g_vals.reshape(-1,1))

    
    def prepare_forward_problem(self, v_2):
        """
        Prepares the forward constraints surrogates and decision variables
        """    
        # load the forward objective
        if self.cfg.solvers.standardised:
            inputs = []
            for i in range(v_2.shape[0]):
                inputs.append(self.standardise_model_decisions([v for v in v_2[i]], self.node))
            inputs = jnp.vstack(inputs)
        else:
            inputs = v_2
        # vmap
        node_constraints =  vmap(partial(lambda x, b: - self.graph.nodes[self.node]["classifier"](x) - b, b=self.graph.nodes[self.node]['constraint_backoff']), in_axes=0, out_axes=0)

        # return the forward surrogates and decision bounds
        return node_constraints, inputs


    
    def standardise_model_decisions(self, decisions, in_node):
        """
        Standardises the decisions
        """
        try:
            mean, std = self.graph.nodes[in_node]['classifier_x_scalar'].mean, self.graph.nodes[in_node]['classifier_x_scalar'].std
            return [(decision - m) / s for i, (m, s, decision) in enumerate(zip([m for m in mean], [s for s in std], decisions)) if i < len(decisions)]
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
        input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
        aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]))
        
        # standardisation of outputs if required
        if cfg.solvers.standardised: succ_inputs[succ] = succ_inputs[succ].at[:].set(standardise_inputs(graph, succ_inputs[succ], succ, jnp.hstack([input_indices, aux_indices]).astype(int)))
        
        # load the standardised bounds
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        ndim = graph.nodes[succ]['n_design_args'] + graph.nodes[succ]['n_input_args'] + graph.graph['n_aux_args']
        
        # get the decision bounds
        if cfg.solvers.standardised: decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        
        decision_bounds = [jnp.delete(bound, np.hstack([input_indices,aux_indices]).astype(int), axis=1) for bound in decision_bounds]
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
    opt_ind = jnp.delete(total_indices, np.concatenate([fix_ind, aux_ind]).astype(int))
    
    # Assign values from x to input_ at positions not in fix_ind and aux_ind
    input_ = input_.at[opt_ind].set(x.squeeze()) 
    
    # Assign values from y to input_ at positions in fix_ind and aux_ind
    if fix_ind.size != 0: input_ = input_.at[fix_ind].set(y[0,:len(fix_ind)])
    if aux_ind.size != 0: input_ = input_.at[aux_ind].set(y[0,len(fix_ind):])
    
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
    feasibility_call = partial(jax_pmap_evaluator, cfg=cfg, graph=graph, node=node)
    
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

    del output_batches, aux_batches, batch_sizes

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

    