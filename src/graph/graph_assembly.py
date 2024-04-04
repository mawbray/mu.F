import networkx as nx
import numpy as np
import queue
import matplotlib.pyplot as plt
from dynamics import deus_wrapper_forward_unit_dynamics
from steady_state import deus_wrapper_forward_unit_continuous

def case_study_uncertain_parameters_dummy(G):
    """
    Add uncertain parameters to the nodes of the graph
    :param G: The graph
    """
    parameter_be = 1.0 # best estimate
    p_sdev = 0. #p.sqrt(0.3)
    np.random.seed(1)
    n_samples_p = 1
    p_samples = np.random.normal(parameter_be, p_sdev, n_samples_p)
    p_samples = [{'c': [p], 'w': 1.0/n_samples_p} for p in p_samples]
    P_BE, P_S = {}, {}

    for node in G.nodes:
        P_BE[node] = [parameter_be]
        P_S[node] = p_samples

    return P_BE, P_S


def case_study_forward_evaluators(cfg, adjacency_matrix):
    """
    Add forward evaluators to the nodes of the graph
    :param G: The graph
    """
    forward_evaluators = {}
    for node in range(len(adjacency_matrix)):
        if cfg.case_study == 'serial_mechanism_batch': forward_evaluators[node] = deus_wrapper_forward_unit_dynamics
        elif cfg.case_study == 'continuous_tableting_operation': forward_evaluators[node] = deus_wrapper_forward_unit_continuous
        else: raise NotImplementedError(f"Case study {cfg.case_study} not implemented")
    
    return forward_evaluators


def case_study_miscellaneous_additions(G, n_input_args, n_design_args, KS_bounds, parameters_best_estimate, parameters_samples):
    """
    Add miscellaneous information to the graph
    :param G: The graph
    :param n_input_args: Dictionary of the number of input arguments associated with each edge
    :param n_design_args: The number of design arguments associated with each node (Dictionary)
    :param KS_bounds: Dictionary of box bounds for the unit KS
    :param parameters_best_estimate: Dictionary of best estimate of parameters
    :param parameters_samples: Dictionary of samples of parameters
    :return: The graph with the miscellaneous information added
    """

    for node in G.nodes:
        # update node features
        G.nodes[node]["n_design_args"] = n_design_args[node]
        G.nodes[node]["KS_bounds"] = KS_bounds[node]
        G.nodes[node]['extendedDS_bounds'] = None
        G.nodes[node]["parameters_best_estimate"] = parameters_best_estimate[node]
        G.nodes[node]["parameters_samples"] = parameters_samples[node]
        G.nodes[node]['fn_evals'] = 0
        n_d = G.nodes[node]['n_design_args']
        # update predecessor - node edge data
        for predec in G.predecessors(node):
            G.edges[predec, node]["n_input_args"] = n_input_args[(predec, node)]
            G.edges[predec, node]["input_indices"] = [n_d + i for i in range(n_input_args[(predec, node)])]
            n_d += n_input_args[(predec, node)]
        G.nodes[node]["n_input_args"] = sum([G.edges[predec, node]["n_input_args"] for predec in G.predecessors(node)])
        

    return G

def case_study_graph_construction(adjacency_matrix, dict_of_forward_eval_fn, dict_of_edge_fn):
    """
    Construct a directed acyclic graph (DAG) from an adjacency matrix and a list of forward evaluators and edge functions
    :param adjacency_matrix: The adjacency matrix of the graph
    :param list_of_forward_eval_fn: A list of forward evaluators for each node
    :param list_of_edge_fn: A list of edge functions for each edge
    :return: The constructed graph
    """

    # Create a directed graph
    G = load_dag_from_adjacency_matrix(adjacency_matrix)
    num_nodes = len(adjacency_matrix) # will be square matrix

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] == 1:
                G[i][j]["edge_fn"] = dict_of_edge_fn[(i, j)]

    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = dict_of_forward_eval_fn[node] 

    return G

def case_study_constraint_addition(G, constraint_dictionary, constraint_args):
    """
    Add constraints to the nodes of the graph
    :param G: The graph
    :param constraint_dictionary: A nested dictionary of constraint function dictionaries
    :param constraint_args: A nested dictionary of constraint arguments
    :return: The graph with constraints added to the nodes
    """
    for node in G.nodes:
        G.nodes[node]["constraints"] = constraint_dictionary[node] # this is a dictionary 
        G.nodes[node]["constraint_args"] = constraint_args # this is a dictionary
    return G


def load_dag_from_adjacency_matrix(adj_matrix):
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G


def allocate_functions_to_nodes(G, node, label, object_allocated):
    G.nodes[node][label] = object_allocated
    return


def allocate_functions_to_edges(G, functions):
    for (i, j), function in zip(G.edges, functions):
        G[i][j]["function"] = functions[(i, j)]
    return


