from abc import ABC
from typing import Iterable, Callable, List
from omegaconf import DictConfig

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, pmap, devices
from functools import partial
from scipy.stats import beta
import jax.scipy.stats as jscp_stats


from constraints.utilities import worker_function, parallelise_batch, determine_batches, create_batches

class constraint_evaluator_base(ABC):
    def __init__(self, cfg, graph, node):
        self.cfg = cfg
        self.graph = graph
        self.node = node

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

    def __call__(self, dynamics_profile):
        return self.evaluate(dynamics_profile)

    def evaluate(self, dynamics_profile):
        """
        Evaluates the constraints
        """
        constraints = self.load_unit_constraints()
        if len(constraints) > 0: 
            constraint_holder = []
            for cons_fn in constraints: # iterate over the constraints that were previously loaded as a dictionary on to the graph
                g = cons_fn(dynamics_profile) # positive constraint value means constraint is satisfied
                if g.ndim < 2: g = g.reshape(-1, 1)
                constraint_holder.append(g)
            return jnp.concatenate(constraint_holder, axis=-1)
        else:
            return None # return None if no unit level constraints are imposed.
        
    def load_unit_constraints(self):
        """
        Loads the constraints from the graph 
        """
        return list(self.graph.nodes[self.node]['constraints'])
    
    def vmap_evaluation(self):
        """
        Vectorizes the the constraints and then loads them back onto the graph
        """
        # get constraints from the graph
        constraints = self.graph.nodes[self.node]['constraints']
        # vectorize each constraint
        cons = (jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg), in_axes=(0), out_axes=0)), in_axes=(1), out_axes=1)) for constraint in constraints)
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
            results = parallelise_batch(solver, device, init_guess)
            for j, result in enumerate(results):
                result_dict[evals + j] = result
            evals += len(results)

        return result_dict



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
        if self.pool != 'jax-pmap':
            self.evaluation_method = self.mp_evaluation
        # shaping function to return to sampler. (depends on the way constraints are defined by the user)
        if cfg.samplers.notion_of_feasibility == 'positive': # i.e. feasible is g(x)>=0
            self.shaping_function = lambda x: -x
        elif cfg.samplers.notion_of_feasibility == 'negative': # i.e. feasible is g(x)<=0
            self.shaping_function = lambda x: x
        else:
            raise ValueError("Invalid notion of feasibility.")
        
    def __call__(self, inputs):
        return self.evaluate(inputs)

    def get_predecessors_inputs(self, inputs):
        """
        Gets the inputs from the predecessors
        """
        pred_inputs = {}
        for pred in self.graph.predecessors(self.node):
            input_indices = self.graph.edges[pred, self.node]['input_indices']
            pred_inputs[pred]= inputs.at[:,input_indices]
        return pred_inputs


    def load_solver(self):
        """
        Loads the solver
        """
        return self.graph.nodes[self.node]['forward_coupling_solver']
    
    def evaluate(self, inputs):
        """
        Evaluates the constraints
        """
        objective, constraints, bounds = self.prepare_forward_problem(inputs)
        solver_object = self.load_solver()  # solver type has been defined elsewhere in the case study/graph construction. 
        pred_fn_evaluations = {}
        # iterate over predecessors and evaluate the constraints
        for pred in self.graph.predecessors(self.node):
            forward_solver = [partial(worker_function, solver=solver_object.from_method(solver_object.logging, solver_object.cfg, solver_object.solver_type, objective[pred][i], bounds[pred][i], constraints[pred][i])) for i in range(inputs.shape[0])] # update the solver, returns a new class specific to the subproblem
            initial_guesses = [solve.initial_guess() for solve in forward_solver]
            pred_fn_evaluations[pred] = self.evaluation_method(forward_solver, initial_guesses, max_devices=self.cfg.max_devices) # TODO make sure this is compatible with format returned by solver

        # reshape the evaluations and just get information about the constraints.
        fn_evaluations = [pred_fn_evaluations[pred]['objective'] for pred in self.graph.predecessors(self.node)]

        return self.shaping_function(jnp.hstack(fn_evaluations))

    
    def prepare_forward_problem(self, inputs):
        """
        Prepares the forward constraints surrogates and decision variables
        """
        forward_constraints = {pred: None for pred in self.graph.predecessors(self.node)}
        forward_bounds = {pred: None for pred in self.graph.predecessors(self.node)}
        forward_objective = {pred: None for pred in self.graph.predecessors(self.node)}

        # get the inputs from the predecessors of the node
        pred_inputs = self.get_predecessors_inputs(inputs)

        for pred in self.graph.predecessors(self.node):
            # standardisation of inputs if required
            if self.cfg.standardisation.forward_coupling: pred_inputs = self.standardise_inputs(pred_inputs, pred)
            # load the forward surrogate
            surrogate = self.graph.edges[pred, self.node]["forward_surrogate"]
            # create a partial function for the optimizer to evaluate
            forward_constraints[pred] = [jit(partial(lambda x, inputs: surrogate(x).reshape(-1,) - inputs.reshape(-1,), inputs=pred_inputs[i,:]) for i in range(pred_inputs.shape[0]))]
            # load the standardised bounds
            decision_bounds = self.graph.nodes[pred]["extendedDS_bounds"].copy()
            if self.cfg.standardisation.forward_coupling: decision_bounds = self.standardise_model_decisions(decision_bounds, pred)
            forward_bounds[pred] = [decision_bounds.copy() for i in range(pred_inputs.shape[0])]
            # load the forward objective
            classifier = self.graph.nodes[pred]["classifier"]
            forward_objective[pred] = [jit(lambda x: classifier(x).squeeze()) for _ in range(pred_inputs.shape[0])]

        # return the forward surrogates and decision bounds
        return forward_objective, forward_constraints, forward_bounds
    
    def standardise_inputs(self, inputs, in_node):
        """
        Standardises the inputs
        """
        mean, std = self.graph.edges[in_node, self.node]['y_scalar'].mean_, self.graph.edges[in_node,self.node]['y_scalar'].std_
        return (inputs - mean) / std
    
    def standardise_model_decisions(self, decisions, in_node):
        """
        Standardises the decisions
        """
        mean, std = self.graph.nodes[in_node]['x_scalar'].mean_, self.graph.nodes[in_node]['x_scalar'].std_
        return [(decision - mean) / std for decision in decisions]
    

class backward_surrogate_constraint(coupling_surrogate_constraint_base):
    """
    A backward surrogate constraint
    - solved with JaxOpt solvers
    - parallelism is provided by jax-pmap
        : may be extended to other solvers e.g. casADi/IPOPT in the future

    TODO - think about how best to pass uncertain parameters.
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node)
        # pool settings
        self.pool = pool
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool == 'jax-pmap':
            self.evaluation_method = self.jax_pmap_evaluation
        if self.pool != 'jax-pmap':
            raise NotImplementedError("parallelisation via other means than jax-pmap is not supported for this constraint type at the moment.")
        # shaping function to return to sampler. (depends on the way constraints are defined by the user)
        if cfg.notion_of_feasibility == 'positive': # i.e. feasible is g(x)>=0
            self.shaping_function = lambda x: -x
        elif cfg.notion_of_feasibility == 'negative': # i.e. feasible is g(x)<=0
            self.shaping_function = lambda x: x
        else:
            raise ValueError("Invalid notion of feasibility.")
    
    @staticmethod
    def from_method(cfg, graph, node, pool):
        return backward_surrogate_constraint(cfg, graph, node, pool)
        

    def get_successor_inputs(self, outputs):
        """
        Gets the inputs from the predecessors
        """
        succ_inputs = {}
        for succ in self.graph.sucessor(self.node):
            output_indices = self.graph.edges[self.node, succ]['output_indices']
            succ_inputs[succ]= outputs.at[:,output_indices]
        return succ_inputs


    def load_solver(self):
        """
        Loads the solver
        """
        return self.graph.nodes[self.node]['backward_coupling_solver']
    
    def evaluate(self, outputs):
        """
        Evaluates the constraints
        """
        objective, bounds = self.prepare_backward_problem(outputs)
        solver_object = self.load_solver()
        succ_fn_evaluations = {}
        # iterate over successors and evaluate the constraints
        for succ in self.graph.successors(self.node):
            forward_solver = [solver_object.from_method(solver_object.logging, solver_object.cfg, solver_object.solver_type, objective[succ][i], bounds[succ][i]) for i in range(outputs.shape[0])]
            initial_guesses = [solve.initial_guess() for solve in forward_solver]
            succ_fn_evaluations[succ] = self.evaluation_method(forward_solver, initial_guesses)

        # reshape the evaluations
        fn_evaluations = [succ_fn_evaluations[succ] for succ in self.graph.successors(self.node)] # TODO make sure this is compatible with format returned by solver

        return self.shaping_function(jnp.hstack(fn_evaluations))
    
    def simple_evaluation(self, solver, initial_guesses, pool):
        """
        Simple evaluation of the constraints
        """

        return solver[0].solve(initial_guesses[0])
    
    def prepare_backward_problem(self, outputs):
        """
        Prepares the forward constraints surrogates and decision variables
        - ouptuts from a nodes unit functions are inputs to the next unit

        """
        backward_bounds = {succ: None for succ in self.graph.successors(self.node)}
        backward_objective = {succ: None for succ in self.graph.successors(self.node)}

        # get the outputs from the successors of the node
        succ_inputs = self.get_successors_inputs(outputs)

        for succ in self.graph.successors(self.node):

            n_d  = self.graph.nodes[succ]['n_design_args']
            input_indices = jnp.array([n_d + input_ for input_ in self.graph.edges[self.node, succ]['input_indices']])
            
            # standardisation of outputs if required
            if self.cfg.standardisation.forward_coupling: succ_inputs = self.standardise_inputs(succ_inputs, succ, input_indices)
            
            # load the standardised bounds
            decision_bounds = self.graph.nodes[succ]["extendedDS_bounds"].copy()
            ndim = decision_bounds[0].shape[0]
            
            # get the decision bounds
            if self.cfg.standardisation.forward_coupling: decision_bounds = self.standardise_model_decisions(decision_bounds, succ)
            
            decision_bounds = [jnp.delete(bound, input_indices) for bound in decision_bounds]
            backward_bounds[succ] = [decision_bounds.copy() for i in range(succ_inputs.shape[0])]

            # load the forward objective
            classifier = self.graph.nodes[succ]["classifier"]
            wrapper_classifier = self.mask_classifier(classifier, n_d, ndim, input_indices)
            backward_objective[succ] = [jit(lambda x: wrapper_classifier(x).squeeze()) for _ in range(succ_inputs.shape[0])]

        # return the forward surrogates and decision bounds
        return backward_objective, backward_bounds
    
    def mask_classifier(self, classifier, nd, ndim, fix_ind):
        """
        Masks the classifier
        - y corresponds to those indices that are fixed
        - x corresponds to those indices that are optimised
        """
        total_indices = jnp.arange(ndim)
        opt_ind = jnp.delete(total_indices, fix_ind)

        @jit
        def masked_classifier(x, y):
            input_ = jnp.zeros(ndim)
            input_ = input_.at[opt_ind].set(x)
            input_ = input_.at[fix_ind].set(y)
            return classifier(input_)
        
        return masked_classifier
    
    def standardise_inputs(self, succ_inputs, out_node, input_indices):
        """
        Standardises the inputs
        """
        mean, std = self.graph.edges[out_node]['y_scalar'].mean_, self.graph.edges[out_node]['y_scalar'].std_
        return (succ_inputs - mean[input_indices]) / std[input_indices]
    
    def standardise_model_decisions(self, decisions, out_node):
        """
        Standardises the decisions
        """
        mean, std = self.graph.nodes[out_node]['x_scalar'].mean_, self.graph.nodes[out_node]['x_scalar'].std_
        return [(decision - mean) / std for decision in decisions]


def jax_pmap_evaluator(outputs, cfg, graph, node, pool):
    """
    p-map constraint evaluation call - called by backward_surrogate_pmap_batch_evaluator
    """
    constraint_evaluator = backward_surrogate_constraint.from_method(cfg, graph, node, pool)
    min_obj = constraint_evaluator.evaluate(outputs)

    return min_obj


def backward_surrogate_pmap_batch_evaluator(outputs, cfg, graph, node, pool):
    """
    Evaluates the constraints on a batch using jax-pmap - called by the backward_constraint_evaluator
    """
    feasibility_call = partial(jax_pmap_evaluator, cfg=cfg, graph=graph, node=node, pool=pool)

    return pmap(feasibility_call, in_axes=(0), out_axes=0,devices=[device for i, device in enumerate(devices('cpu')) if i<outputs.shape[0]])(outputs)
 

def backward_constraint_evaluator(outputs, cfg, graph, node, pool):
    """
    Evaluates the constraints using jax-pmap - this is what should be called

    Syntax: 
        call: method_(outputs, cfg, graph, node, pool)

    """
    max_devices = cfg.max_devices
    batch_sizes, remainder = determine_batches(outputs.shape, max_devices)
    # get batches of outputs
    output_batches = create_batches(batch_sizes, outputs)
    # evaluate the constraints
    results = []
    for i, output_batch in enumerate(output_batches):
        results.append(backward_surrogate_pmap_batch_evaluator(output_batch, cfg, graph, node, pool))
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

    