"""
Utility functions for constraints
"""
from typing import Callable
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.stats import beta

def standardise_inputs(graph, succ_inputs, out_node, input_indices):
    """
    Standardises the inputs
    """
    if out_node is not None:
        standardiser = graph.nodes[out_node]['classifier_x_scalar']
    else:
        standardiser = graph.graph['classifier_x_scalar']

    if standardiser is None: return succ_inputs
    else:
        mean, std = standardiser.mean, standardiser.std
        return (succ_inputs - mean[input_indices].reshape(1,-1)) / std[input_indices].reshape(1,-1)


def standardise_model_decisions(graph, decisions, out_node):
    """
    Standardises the decisions
    """
    if out_node is None:
        standardiser = graph.graph['classifier_x_scalar']
    else:
        standardiser = graph.nodes[out_node]['classifier_x_scalar']
    if standardiser is None: return decisions
    else:
        mean, std = standardiser.mean, standardiser.std
        return [(decision - mean[:].reshape(1,-1)) / std[:].reshape(1,-1) for decision in decisions]
    

def mask_classifier(classifier: Callable, ndim, fix_ind, aux_ind):
    """
    Masks the classifier
    - y corresponds to those indices that are fixed
    - x corresponds to those indices that are optimised
    """

    def masked_classifier(x, y):
        input_ = construct_input(x, y, fix_ind, aux_ind, ndim)
        return classifier(input_.reshape(1,-1)).squeeze()
    
    return jit(masked_classifier)


def construct_input(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        fix_ind: jnp.ndarray, 
        aux_ind: jnp.ndarray, 
        ndim: int
    ) -> jnp.ndarray:

    """
    Constructs the input to the classifier
    - y corresponds to those indices that are fixed
        assumed to be of shape (1, len(fix_ind) + len(aux_ind))
    - x corresponds to those indices that are optimised
        assumed to be of shape (len(decision_vars), ), such that 
        len(x) + len(y) = ndim
    - fix_ind are the indices of the fixed (design or input) variables
    - aux_ind are the indices of the fixed auxiliary variables 
    - ndim is the total number of dimensions 
      assumed to be len(x) + len(y)
    :return: the constructed input of shape (ndim, )
    """
    # Initialize input_ with zeros or any placeholder value
    input_ = jnp.zeros(ndim)
    
    # Create a mask for positions not in fix_ind and aux_ind
    total_indices = np.arange(ndim)
    opt_ind = np.delete(total_indices, np.concatenate([fix_ind, aux_ind]).astype(int))
    
    # Assign values from x to input_ at positions not in fix_ind and aux_ind
    input_ = input_.at[opt_ind].set(x.squeeze()) 
    
    # Assign values from y to input_ at positions in fix_ind and aux_ind
    if (y.shape[1] >= len(fix_ind)): # note that this will not be the case if graph_wide_problem is solved at a Node
        if (fix_ind.size != 0): input_ = input_.at[fix_ind].set(y[0,:len(fix_ind)])
        if aux_ind.size != 0: input_ = input_.at[aux_ind].set(y[0,len(fix_ind):])
        
    return input_

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

    