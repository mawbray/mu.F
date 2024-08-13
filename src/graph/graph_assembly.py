from abc import ABC
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt





class graph_constructor_base(ABC):
    def __init__(self, cfg, adjacency_matrix):
        self.cfg = cfg
        self.adjacency_matrix = adjacency_matrix
        self.G = load_dag_from_adjacency_matrix(adjacency_matrix)
        self.num_nodes = len(adjacency_matrix)

    def get_graph(self):
        return self.G.copy()
    
    def add_arg_to_nodes(self, arg_name, arg_value):
        for node in self.G.nodes:
            self.G.nodes[node][arg_name] = arg_value[node]
        return
    
    def add_arg_to_edges(self, arg_name, arg_value):
        for (i, j) in self.G.edges:
            self.G.edges[i,j][arg_name] = arg_value[(i, j)]
        return
    
    def add_node_object(self, node, node_object, node_object_name):
        self.G.nodes[node][node_object_name] = node_object
        return
    
    def add_edge_object(self, i, j, edge_object, edge_object_name):
        self.G.edges[i, j][edge_object_name] = edge_object
        return

    

class graph_constructor(graph_constructor_base):
    def __init__(self, cfg, adjacency_matrix):
        super().__init__(cfg, adjacency_matrix)
     
    
    def add_n_input_args(self, n_input_args):
        for (i, j) in self.G.edges:
            self.G.edges[i,j]["n_input_args"] = n_input_args[f'({i},{j})']
        for node in self.G.nodes:
            self.G.nodes[node]["n_input_args"] = sum([self.G.edges[predec, node]["n_input_args"] for predec in self.G.predecessors(node)])
        return
    
   
    def add_input_indices(self):
        for node in self.G.nodes:
            n_d = 0
            for predec in self.G.predecessors(node):
                self.G.edges[predec, node]["input_indices"] = [n_d + i for i in range(self.G.edges[predec, node]["n_input_args"])]
                n_d += self.G.edges[predec, node]["n_input_args"]
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



def load_dag_from_adjacency_matrix(adj_matrix):
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G



