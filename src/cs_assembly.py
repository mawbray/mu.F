
from unit_evaluators.constructor import unit_evaluation


def case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, constraint_args):
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
    G.add_arg_to_nodes('n_design_args', cfg.n_design_args)
    G.add_arg_to_nodes('KS_bounds', cfg.KS_bounds)
    G.add_arg_to_nodes('parameters_best_estimate', cfg.parameters_best_estimate)
    G.add_arg_to_nodes('parameters_samples', cfg.parameters_samples)
    G.add_arg_to_nodes('fn_evals', cfg.fn_evals)
    G.add_arg_to_nodes('unit_op', cfg.unit_op)
    G.add_arg_to_nodes('unit_params_fn', cfg.unit_params_fn)
    G.add_arg_to_nodes('extendedDS_bounds', cfg.extendedDS_bounds)
    G.add_arg_to_nodes('constraints', constraint_dictionary)
    G.add_arg_to_nodes('constraint_args', constraint_args)


    # add miscellaneous information to the graph
    G.add_n_input_args(cfg.n_input_args)
    G.add_input_indices()

    # add edge properties to the graph
    G.add_arg_to_edges('edge_fn', dict_of_edge_fn)

    graph = G.get_graph()

    for node in graph.nodes:
        G.add_node_object(node, unit_evaluation(cfg, graph, node), "forward_evaluator")


    return G