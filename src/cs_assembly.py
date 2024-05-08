
from unit_evaluators.constructor import unit_evaluation
from constraints.functions import CS_holder
from graph.graph_assembly import graph_constructor
from graph.methods import CS_edge_holder, vmap_CS_edge_holder
from constraints.solvers.constructor import solver_construction

import logging

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

    # Call the case_study_allocation function
    G = case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers=solver_constructor(cfg, G))

    return G.get_graph()



def case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers):
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
    G.add_arg_to_nodes('KS_bounds', cfg.case_study.KS_bounds)
    G.add_arg_to_nodes('parameters_best_estimate', cfg.case_study.parameters_best_estimate)
    G.add_arg_to_nodes('parameters_samples', cfg.case_study.parameters_samples)
    G.add_arg_to_nodes('fn_evals', cfg.case_study.fn_evals)
    G.add_arg_to_nodes('unit_op', cfg.case_study.unit_op)
    G.add_arg_to_nodes('unit_params_fn', cfg.case_study.unit_params_fn)
    G.add_arg_to_nodes('extendedDS_bounds', cfg.case_study.extendedDS_bounds)
    G.add_arg_to_nodes('constraints', constraint_dictionary)
    G.add_arg_to_nodes('forward_coupling_solver', solvers['forward_coupling_solver'])
    G.add_arg_to_nodes('backward_coupling_solver', solvers['backward_coupling_solver'])


    # add miscellaneous information to the graph
    G.add_n_input_args(cfg.case_study.n_input_args)
    G.add_input_indices()

    # add edge properties to the graph
    G.add_arg_to_edges('edge_fn', dict_of_edge_fn)

    graph = G.get_graph()

    for node in graph.nodes:
        G.add_node_object(node, unit_evaluation(cfg, graph, node), "forward_evaluator")


    return G


def solver_constructor(cfg, G):

    return  {'forward_coupling_solver': {node: solver_construction for node in G.G.nodes },  # if G.G.in_degree()[node] > 0 (this is better, but raises errors downstream, so we'll roll with it for now) 
            'backward_coupling_solver': {node: solver_construction for node in G.G.nodes}}# if G.G.out_degree()[node] > 0 (this is better, but raises errors downstream, so we'll roll with it for now) 