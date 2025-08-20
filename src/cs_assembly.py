
from unit_evaluators.constructor import unit_evaluation, post_process_evaluation
from constraints.functions import CS_holder, post_process_visualiser
from graph.graph_assembly import graph_constructor
from graph.methods import CS_edge_holder, vmap_CS_edge_holder
from constraints.solvers.constructor import solver_construction
from constraints.solvers.surrogate.surrogate import surrogate
from constraints.constructor import constraint_evaluator
from post_processes.constructor import post_process
from unit_evaluators.utils import arrhenius_kinetics_fn, arrhenius_kinetics_fn_2
from visualisation.visualiser import visualiser

from functools import partial
from itertools import chain
import logging
import jax.numpy as jnp
import numpy as np
import pandas as pd

def case_study_constructor(cfg):
    """
    Construct the case study graph
    :param cfg: The configuration object
    :return: The graph construction object
    """

    # Create a sample constraint dictionary
    constraint_dictionary = CS_holder[cfg.case_study.case_study]

    # create edge functions
    if cfg.case_study.vmap_evaluations:
        dict_of_edge_fn = vmap_CS_edge_holder[cfg.case_study.case_study]
    else:
        dict_of_edge_fn = CS_edge_holder[cfg.case_study.case_study]

    # Create a graph constructor object
    G = graph_constructor(cfg, cfg.case_study.adjacency_matrix)

    # construct dummy dataframe for initial forward pass
    init_df_samples = pd.DataFrame({col: np.zeros((2,)) for i,col in enumerate(cfg.case_study.design_space_dimensions)})

    # Call the case_study_allocation function
    G = case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers=solver_constructor(cfg, G), unit_params_fn=unit_params_fn(cfg, G), initial_forward_pass=init_df_samples)

    return G.get_graph()



def case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers, unit_params_fn, initial_forward_pass):
    """
    Add miscellaneous information to the graph
    :param G: The graph constructor
    :param n_input_args: Dictionary of the number of input arguments associated with each edge
    :param n_design_args: The number of design arguments associated with each node (Dictionary)
    :param KS_bounds: Dictionary of box bounds for the unit KS
    :param parameters_best_estimate: Dictionary of best estimate of parameters
    :param parameters_samples: Dictionary of samples of parameters
    :return: The graph construction object with the miscellaneous information added
    """

    # add nodes properties to the graph
    G.add_arg_to_nodes('n_design_args', cfg.case_study.n_design_args)
    G.add_arg_to_nodes('n_theta', cfg.case_study.n_theta)
    G.add_arg_to_nodes('KS_bounds', cfg.case_study.KS_bounds.design_args)
    G.add_arg_to_nodes('parameters_best_estimate', cfg.case_study.parameters_best_estimate)
    G.add_arg_to_nodes('parameters_samples', cfg.case_study.parameters_samples)
    G.add_arg_to_nodes('fn_evals', cfg.case_study.fn_evals)
    G.add_arg_to_nodes('unit_op', cfg.case_study.unit_op)
    G.add_arg_to_nodes('unit_params_fn', unit_params_fn)
    G.add_arg_to_nodes('extendedDS_bounds', cfg.case_study.extendedDS_bounds)
    G.add_arg_to_nodes('constraints', constraint_dictionary)
    
    G.add_arg_to_nodes('forward_coupling_solver', solvers['forward_coupling_solver'])
    G.add_arg_to_nodes('backward_coupling_solver', solvers['backward_coupling_solver'])
    if cfg.method != 'decomposition_constraint_tuner':
        b_off = [0 for _ in range(len(cfg.case_study.adjacency_matrix))]
        G.add_arg_to_nodes('constraint_backoff', b_off)

    # add miscellaneous information to the graph
    G.add_n_input_args(cfg.case_study.n_input_args)
    G.add_n_aux_args(cfg.case_study.n_aux_args)
    G.add_input_aux_indices()

    # add the args to the graph 
    G.add_arg_to_graph('aux_bounds', cfg.case_study.KS_bounds.aux_args)
    G.add_arg_to_graph('n_aux_args', cfg.case_study.global_n_aux_args)
    G.add_arg_to_graph('initial_forward_pass', initial_forward_pass)
    G.add_arg_to_graph('n_design_args', sum(cfg.case_study.n_design_args))
    G.add_arg_to_graph('bounds', list(chain.from_iterable(cfg.case_study.KS_bounds.design_args + cfg.case_study.KS_bounds.aux_args)))
    G.add_arg_to_graph('classifier_x_scalar', None) # initialisation
    G.add_arg_to_graph('post_process_classifier', lambda x: jnp.sum(x)) # scalarise input as dummy function for jit tracing.
    G.add_arg_to_graph('post_process_lower_classifier', lambda x: jnp.sum(x))
    if cfg.reconstruction.post_process:
        # post processing
        G.add_arg_to_nodes('post_process_constraints', post_process_visualiser[cfg.case_study.case_study])
        G.add_arg_to_graph('post_process', post_process)
        G.add_arg_to_graph('post_process_training_methods', surrogate)
        G.add_arg_to_graph('post_process_solver_methods', {'upper_level_solver': constraint_evaluator, 'lower_level_solver': constraint_evaluator})
        G.add_arg_to_graph('post_process_decision_indices', cfg.reconstruction.post_process_decision_indices)
        G.add_arg_to_graph('solve_post_processing_problem', False) # overwritten in the post_process function
        G.add_arg_to_graph('post_process_solution_evaluator', partial(post_process_evaluation, constraint_evaluator=constraint_evaluator))
        G.add_arg_to_graph('post_process_solution_visualiser', visualiser) 
    # add edge properties to the graph
    G.add_arg_to_edges('edge_fn', dict_of_edge_fn)
    # add the auxiliary filters to the graph
    G.add_arg_to_edges('aux_filter', aux_filter(cfg, G))

    graph = G.get_graph()

    for node in graph.nodes:
        G.add_node_object(node, unit_evaluation(cfg, graph, node), "forward_evaluator")

    return G


def unit_params_fn(cfg, G):

    if cfg.case_study.case_study == 'batch_reaction_network' or (cfg.case_study.case_study == 'serial_mechanism_batch'):
        return {node: partial(arrhenius_kinetics_fn_2,Ea=jnp.array(cfg.model.arrhenius.EA[node]), R=jnp.array(cfg.model.arrhenius.R)) for node in G.G.nodes}
    elif cfg.case_study.case_study == 'serial_mechanism_batch':
        return {node: partial(arrhenius_kinetics_fn,Ea=jnp.array(cfg.model.arrhenius.EA[node]), A=jnp.array(cfg.model.arrhenius.A[node]), R=jnp.array(cfg.model.arrhenius.R)) for node in G.G.nodes}
    elif cfg.case_study.case_study in ['tablet_press', 'convex_estimator', 'estimator', 'convex_underestimator', 'affine_study']:
        return {node: lambda x, y: jnp.empty((0,)) for node in G.G.nodes}
    else :
        raise ValueError('Invalid case study')
    

def aux_filter(cfg, G):
    if G.G.graph['n_aux_args'] == 0:
        return {edge: lambda x: x for edge in G.G.edges}
    else:
        return {edge: lambda x: [x[0][:,:-G.G.graph['n_aux_args']], x[1][:,:-G.G.graph['n_aux_args']]] for edge in G.G.edges}

def solver_constructor(cfg, G):

    return  {'forward_coupling_solver': {node: solver_construction for node in G.G.nodes },  # if G.G.in_degree()[node] > 0 (this is better, but raises errors downstream, so we'll roll with it for now) 
            'backward_coupling_solver': {node: solver_construction for node in G.G.nodes}}# if G.G.out_degree()[node] > 0 (this is better, but raises errors downstream, so we'll roll with it for now) 