from abc import ABC
from typing import Iterable, Callable, List
from omegaconf import DictConfig

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from scipy.stats import beta
import jax.scipy.stats as jscp_stats
from multiprocessing import Pool

from utilities import worker_function, parallelise_batch, determine_batch_size, create_batches

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

    def __init__(self, cfg, graph, node):
        """
        Initializes
        """
        super().__init__(cfg, graph, node)
        if cfg.vmap_constraint_evaluation:
            self.vmap_evaluation()

    def evaluate(self, dynamics_profile):
        """
        Evaluates the constraints
        """
        constraints = self.load_unit_constraints()
        if len(constraints) > 0: 
            constraint_holder = []
            for cons_fn in constraints:
                g = cons_fn(dynamics_profile, self.cfg) # positive constraint value means constraint is satisfied
                if g.ndim < 2: g = g.reshape(-1, 1)
                constraint_holder.append(g)
            return jnp.hstack(constraint_holder)
        else:
            return None # return None if no unit level constraints are imposed.
        
    def load_unit_constraints(self):
        """
        Loads the constraints from the graph 
        """
        return list(self.graph.nodes[self.node]['constraints'].values())
    
    def vmap_evaluation(self):
        """
        Vectorizes the the constraints and then loads them back onto the graph
        """
        # get constraints from the graph
        constraints = self.graph.nodes[self.node]['constraints'].copy()
        # vectorize each constraint
        for key, constraint in constraints.items():
            constraints[key] = jit(vmap(constraint, in_axes=(0, None), out_axes=0))
        # load the vectorized constraints back onto the graph
        self.graph.nodes[self.node]['constraints'] = constraints

        return 

class forward_surrogate_constraint_base(constraint_evaluator_base):
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
        workers, remainder = determine_batch_size(len(solver), max_devices)
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



class forward_surrogate_constraint(forward_surrogate_constraint_base):
    """
    A forward surrogate constraint
    - solved using casadi interface with jax and IPOPT
    - parallelism is provided by multiprocessing pool
        : may be extended to jax-pmap in the future if someone develops a nice nlp solver in jax
    """
    def __init__(self, cfg, graph, node, pool):
        super().__init__(cfg, graph, node)
        # pool settings
        self.pool = pool
        if self.pool is None: 
            raise Warning("No multiprocessing pool provided. Forward surrogate constraints will be evaluated sequentially.")
        if self.pool is 'jax-pmap':
            raise NotImplementedError("jax-pmap is not supported for this constraint type at the moment.")
        if self.pool != 'jax-pmap':
            self.evaluation_method = self.mp_evaluation
        # shaping function to return to sampler. (depends on the way constraints are defined by the user)
        if cfg.notion_of_feasibility == 'positive': # i.e. feasible is g(x)>=0
            self.shaping_function = lambda x: -x
        elif cfg.notion_of_feasibility == 'negative': # i.e. feasible is g(x)<=0
            self.shaping_function = lambda x: x
        else:
            raise ValueError("Invalid notion of feasibility.")

    def get_predecessors_inputs(self, inputs):
        """
        Gets the inputs from the predecessors
        """
        pred_inputs = {}
        for pred in self.graph.predecessors(self.node):
            input_indices = self.graph.edges[pred, self.node]['input_indices']
            pred_inputs[pred]= inputs[:,input_indices]
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

        fn_evaluations = self.evaluation_method(constraints, objective, bounds, [self.cfg for _ in range(inputs.shape[0])], self.pool)

        return self.shaping_function(jnp.array(fn_evaluations))

    
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
