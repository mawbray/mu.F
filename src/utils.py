import pickle
from abc import ABC
import jax.numpy as jnp
from typing import List

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
    
        
    

class data_processing(ABC):
    def __init__(self, dataset_object):
        self.dataset_object = dataset_object
        self.d = dataset_object.d
        self.p = dataset_object.p
        self.y = dataset_object.y
        self.check_data()

    def check_data(self):
        n_d = len(self.d)
        assert n_d == len(self.p) == len(self.y), "The number of design, parameters and outputs objects within the data object should be the same, currently {} n_d, {} n_p and {} n_o".format(n_d, len(self.p), len(self.y))

    def get_data(self):
        for i in range(len(self.d)):
            yield self.d[i], self.p[i], self.y[i]

    def transform_data_to_matrix(self, edge_fn):
        """
        Dealing with the data in a matrix format
         - This is useful for training neural networks
        """
        data = self.get_data()
        data_store_X, data_store_Y = []
        for d, p, y in data:
            X, Y = [], []
            for i in range(p.shape[0]):
                X.append(jnp.hstack([d, jnp.repeat(p[i],d.shape[0], axis=0)]))
                y_edge = edge_fn(y)
                Y.append(y_edge[:,i,:].reshape(d.shape[0],-1))
            X = jnp.vstack(X)
            Y = jnp.vstack(Y)
            data_store_X.append(X)
            data_store_Y.append(Y)

        return jnp.vstack(data_store_X), jnp.vstack(data_store_Y)
    

class feasibility_base(ABC):
    def __init__(self, dataset_X, dataset_Y, cfg):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
        self.cfg = cfg
        if self.cfg.formulation == 'probabilistic':
            self.feasible_function = self.probabilistic_feasibility
        elif self.cfg.formulation == 'deterministic':
            self.feasible_function = self.deterministic_feasibility
        else:
            raise ValueError(f"Formulation {self.cfg.formulation} not recognised. Please use 'probabilistic' or 'deterministic'.")
    
    def probabilistic_feasibility(self, X, Y):
        """
        Method to evaluate the probabilistic feasibility of the data
        """
        pass

    def deterministic_feasibility(self, X, Y):
        """
        Method to evaluate the deterministic feasibility of the data
        """
        pass


class apply_feasibility(feasibility_base):
    def __init__(self, dataset_X, dataset_Y, cfg):
        super().__init__(dataset_X, dataset_Y, cfg)

    def get_feasible(self):
        return self.feasible_function(self.dataset_X, self.dataset_Y)

    def probabilistic_feasibility(self, X, Y, return_indices=True):
        """
        Method to evaluate the probabilistic feasibility of the data
        """
        select_cond = jnp.where(Y >= self.cfg.probability_level, 1, 0)
        return X[select_cond.squeeze(), :], Y[select_cond.squeeze(), :]

    def deterministic_feasibility(self, X, Y, return_indices=True):
        """
        Method to evaluate the deterministic feasibility of the data
        """
        if self.cfg.notion_of_feasibility == 'positive':
            select_cond = jnp.max(Y, axis=-1)  >= 0 
        else:
            select_cond = jnp.max(Y, axis=-1)  <= 0  
        if not return_indices:
            return X[select_cond.squeeze(), :], Y[select_cond.squeeze(), :]
        else:
            return  X[select_cond.squeeze(), :], Y[select_cond.squeeze(), :], select_cond.squeeze()
