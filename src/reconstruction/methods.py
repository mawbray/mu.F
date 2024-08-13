
import pickle 
import numpy as np



def construct_cartesian_product_of_live_sets(graph):
    """
    Construct the cartesian product of the live sets
    :param graph: The graph
    :param cfg: The configuration
    :return: The graph
    """
    # Construct the cartesian product of the live sets
    unit_ls = {}
    for node in graph.nodes:
        n_d = graph.nodes[node]['n_design_args']
        rng = np.random.default_rng()
        lset = np.copy(graph.nodes[node]["live_set_inner"][:,:n_d]).reshape(-1, n_d)
        rng.shuffle(lset, axis = 0)
        unit_ls[node] = np.copy(lset)

    return unit_ls


def save_graph(G, filename):
    """
    Save the graph to a file
    :param G: The graph
    :param cfg: The configuration
    :return: None
    """
    # dump everything not needed 
    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = None # drop forward evaluators
        for predec in G.predecessors(node):
            G.edges[predec, node]["edge_fn"] = None # drop edge functions
            G.edges[predec, node]["forward_surrogate"] = None # drop forward surrogates
        G.nodes[node]["constraints"] = None   # drop constraint functions (all of these are non-serializable.)
        G.nodes[node]['classifier'] = None 

    
    # Save the graph to a file

    with open(filename, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_graph(filename):
    """
    Load the graph from a file
    :param filename: The filename
    :return: The graph
    """
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G

def overwrite_graph(G, blank_graph_object):
    """
    Overwrite the graph
    :param G: The graph
    :blank_graph_object: The graph
    :return: graph
    """
    # dump everything not needed 
    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = blank_graph_object.nodes[node]["forward_evaluator"] # drop forward evaluators
        for predec in G.predecessors(node):
            G.edges[predec, node]["edge_fn"] = blank_graph_object.edges[predec, node]["edge_fn"] # drop edge functions
        G.nodes[node]["constraints"] = blank_graph_object.nodes[node]["constraints"]   # drop constraint functions (all of these are non-serializable.)
        
    return G    