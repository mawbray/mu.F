import pickle
from abc import ABC
import jax.numpy as jnp

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


class dataset_object(ABC):
    def __init__(self, d, p, y):
        self.input_rank = len(d.shape)
        self.output_rank = len(y.shape)
        self.d = [d if self.input_rank >= 2 else d.expand_dims(axis=-1)]
        self.p = [p if self.input_rank >= 2 else p.expand_dims(axis=-1)] 
        self.y = [y if self.output_rank >=2 else y.expand_dims(axis=-1)]
        

    def add(self, d_in, p_in, y_in):
        self.d.append(d_in if self.input_rank >= 2 else d_in.expand_dims(axis=-1))
        self.p.append(p_in if self.input_rank >= 2 else p_in.expand_dims(axis=-1))
        self.y.append(y_in if self.output_rank >=2 else y_in.expand_dims(axis=-1))
        return 
    