from abc import ABC
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#from dynamics import deus_wrapper_forward_unit_dynamics
#from steady_state import deus_wrapper_forward_unit_continuous
from src.unit_evaluators.constructor import unit_evaluation


class graph_constructor_base(ABC):
    def __init__(self, cfg, adjacency_matrix):
        self.cfg = cfg

    def get_graph(self):
        raise NotImplementedError
    
    def add_n_design_args(self):
        raise NotImplementedError
    
    def add_n_input_args(self):
        raise NotImplementedError
    
    def add_KS_bounds(self):
        raise NotImplementedError

    def add_parameters_best_estimate(self):
        raise NotImplementedError
    
    def add_parameters_samples(self):
        raise NotImplementedError

    def add_constraints(self):
        raise NotImplementedError

    def add_constraint_args(self):
        raise NotImplementedError

    def add_extendedDS_bounds(self):
        raise NotImplementedError

    def add_fn_evals(self):
        raise NotImplementedError
    
    def add_forward_evaluator(self):
        raise NotImplementedError
    
    def add_edge_fn(self):
        raise NotImplementedError

    def add_input_indices(self):
        raise NotImplementedError

    def add_unit_op(self):
        raise NotImplementedError

    def add_unit_params_fn(self):
        raise NotImplementedError

    def add_forward_surrogate(self):
        raise NotImplementedError
    
    def add_constraint_surrogate(self):
        raise NotImplementedError
    

class graph_constructor(graph_constructor_base):
    def __init__(self, cfg, adjacency_matrix):
        super().__init__(cfg, adjacency_matrix)
        self.G = load_dag_from_adjacency_matrix(adjacency_matrix)
        self.num_nodes = len(adjacency_matrix)

    def get_graph(self):
        return self.G
    
    def add_n_design_args(self, n_design_args):
        for node in self.G.nodes:
            self.G.nodes[node]["n_design_args"] = n_design_args[node]
        return

    def add_n_input_args(self, n_input_args):
        for (i, j) in self.G.edges:
            self.G.edges[i,j]["n_input_args"] = n_input_args[(i, j)]
        for node in self.G.nodes:
            G.nodes[node]["n_input_args"] = sum([G.edges[predec, node]["n_input_args"] for predec in G.predecessors(node)])
        return
    
    def add_KS_bounds(self, KS_bounds):
        for node in self.G.nodes:
            self.G.nodes[node]["KS_bounds"] = KS_bounds[node]
        return
    
    def add_parameters_best_estimate(self, parameters_best_estimate):
        for node in self.G.nodes:
            self.G.nodes[node]["parameters_best_estimate"] = parameters_best_estimate[node]
        return
    
    def add_parameters_samples(self, parameters_samples):
        for node in self.G.nodes:
            self.G.nodes[node]["parameters_samples"] = parameters_samples[node]
        return
    
    def add_constraints(self, constraint_dictionary):
        for node in self.G.nodes:
            self.G.nodes[node]["constraints"] = constraint_dictionary[node]
        return
    
    def add_constraint_args(self, constraint_args):
        for node in self.G.nodes:
            self.G.nodes[node]["constraint_args"] = constraint_args
        return

    def add_extendedDS_bounds(self, extendedDS_bounds):
        for node in self.G.nodes:
            self.G.nodes[node]["extendedDS_bounds"] = extendedDS_bounds[node]
        return
    
    def add_fn_evals(self, fn_evals):
        for node in self.G.nodes:
            self.G.nodes[node]["fn_evals"] = fn_evals[node]
        return
    
    def add_forward_evaluator(self, forward_evaluator):
        for node in self.G.nodes:
            self.G.nodes[node]["forward_evaluator"] = forward_evaluator[node]
        return
    
    def add_edge_fn(self, edge_fn):
        for (i, j) in self.G.edges:
            self.G.edges[i,j]["edge_fn"] = edge_fn[(i, j)]
        return
    
    def add_input_indices(self):
        for node in G.nodes:
            n_d = G.nodes[node]["n_design_args"]
            for predec in G.predecessors(node):
                G.edges[predec, node]["input_indices"] = [n_d + i for i in range(G.edges[predec, node]["n_input_args"])]
                n_d += G.edges[predec, node]["n_input_args"]

        return
    
    def add_unit_op(self, unit_op):
        for node in self.G.nodes:
            self.G.nodes[node]["unit_op"] = unit_op[node]
        return
    
    def add_unit_params_fn(self, unit_params_fn):
        for node in self.G.nodes:
            self.G.nodes[node]["unit_params_fn"] = unit_params_fn[node]
        return
    
    def add_forward_surrogate(self, forward_surrogate):
        for node in self.G.nodes:
            self.G.nodes[node]["forward_surrogate"] = forward_surrogate[node]
        return

    def add_constraint_surrogate(self, constraint_surrogate):
        for node in self.G.nodes:
            self.G.nodes[node]["constraint_surrogate"] = constraint_surrogate[node]
        return
    
    def add_unit_evaluator(self, unit_evaluator):
        for node in self.G.nodes:
            self.G.nodes[node]["unit_evaluator"] = unit_evaluator[node]
        return
    
    
    




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




def case_study_miscellaneous_additions(G, n_input_args, n_design_args, KS_bounds, parameters_best_estimate, parameters_samples):
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

    G.add_n_design_args(n_design_args)
    G.add_n_input_args(n_input_args)
    G.add_KS_bounds(KS_bounds)
    G.add_parameters_best_estimate(parameters_best_estimate)
    G.add_parameters_samples(parameters_samples)
    G.add_input_indices()

    # TODO initialise the following
    # G.nodes[node]['extendedDS_bounds'] = None
    # G.nodes[node]['fn_evals'] = 0
    # n_d = G.nodes[node]['n_design_args']

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
        # TODO update initialisation of unit evaluation to operate on the graph object.
        G.nodes[node]["forward_evaluator"] = unit_evaluation(cfg, )

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


