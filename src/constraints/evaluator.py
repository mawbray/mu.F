from abc import ABC
import jax.numpy as jnp
from jax import vmap, jit

from functools import partial
from scipy.stats import beta
import jax.scipy.stats as jscp_stats



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


class forward_surrogate_constraint(constraint_evaluator_base):
    """
    A forward surrogate constraint evaluator
    """
    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def load_solver(self):
        """
        Loads the solver
        """
        return self.graph.nodes[self.node]['forward_coupling_solver']
    
    def evaluate(self, inputs):
        """
        Evaluates the constraints
        """
        constraints = self.load_unit_constraints()
        if len(constraints) > 0: 
            constraint_holder = []
            for cons_fn in constraints:
                g = cons_fn(inputs, self.cfg)
                if g.ndim < 2: g = g.reshape(-1, 1)
    
    def prepare_forward_surrogate(self, inputs):
        """
        Prepares the forward constraints
        """
        forward_surrogates = []
        for pred in self.graph.predecessors(self.node):
            if self.cfg.standardisation.forward_coupling:
                inputs = self.standardise_inputs(inputs, pred)
                surrogate = self.graph.edges[pred, self.node]["forward_surrogate"]
                def surrogate_fn(x, inputs):
                    return surrogate(x).reshape(-1,) - inputs.reshape(-1,)
                forward_surrogates.append(jit(partial(surrogate_fn, inputs=inputs)))


        decisions = self.standardise_model_decisions(decisions, pred)

        return self.evaluate(inputs)
    
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
        return (decisions - mean) / std
        


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
