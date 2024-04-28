import pickle

def save_graph(G, mode):
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
    filename = f"graph_{mode}.pickle"
    with open(filename, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return