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
        self.p_rank = len(p.shape)
        self.output_rank = len(y.shape)

        d = d if self.input_rank > 1 else jnp.expand_dims(d,axis=-1)
        p = p if self.p_rank > 1 else jnp.expand_dims(p,axis=-1)
        y = y if self.output_rank >1 else jnp.expand_dims(y,axis=-1)

        self.input_rank = len(d.shape)
        self.p_rank = len(p.shape)
        self.output_rank = len(y.shape)

        self.d = [d if self.input_rank > 2 else jnp.expand_dims(d, axis=-1)]
        self.p = [p if self.p_rank > 2 else jnp.expand_dims(p,axis=-1)] 
        self.y = [y if self.output_rank >2 else jnp.expand_dims(y,axis=-1)]
        

    def add(self, d_in, p_in, y_in):
        input_rank = len(d_in.shape)
        p_rank = len(p_in.shape)
        output_rank = len(y_in.shape)

        d_in = d_in if input_rank > 1 else jnp.expand_dims(d,axis=-1)
        p_in = p_in if p_rank > 1 else jnp.expand_dims(p,axis=-1)
        y_in = y_in if output_rank >1 else jnp.expand_dims(y,axis=-1)

        input_rank = len(d_in.shape)
        p_rank = len(p_in.shape)
        output_rank = len(y_in.shape)
        

        self.d.append(d_in if input_rank > 2 else jnp.expand_dims(d_in,axis=-1))
        self.p.append(p_in if input_rank > 2 else jnp.expand_dims(p_in,axis=-1))
        self.y.append(y_in if output_rank >2 else jnp.expand_dims(y_in,axis=-1))
        return 
    

class dataset(ABC):
    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else jnp.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else jnp.expand_dims(y, axis=-1)
            
        
    

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
        data_store_X, data_store_Y = [], []
        for d, p, y in data:
            X, Y = [], []
            if p.ndim<2: p=p.reshape(1,-1)
            if d.ndim<2: d=d.reshape(1,-1)
            if p.ndim < 3: p = jnp.expand_dims(p, axis=1)
            if d.ndim < 3: d = jnp.expand_dims(d, axis=1)

            y_edge = edge_fn(y)
            if y_edge.ndim < 3: y_edge = jnp.expand_dims(y_edge, axis=-1)

            for i in range(p.shape[0]):
                X.append(jnp.hstack([d.reshape((d.shape[0], d.shape[1])), jnp.repeat(p[i].reshape(1,-1),d.shape[0], axis=0)]))
                Y.append(y_edge[:,i,:].reshape(d.shape[0],-1))
                
            X = jnp.vstack(X)
            Y = jnp.vstack(Y)
            data_store_X.append(X)
            data_store_Y.append(Y)

        return jnp.vstack(data_store_X), jnp.vstack(data_store_Y)
    

class feasibility_base(ABC):
    def __init__(self, dataset_X, dataset_Y, cfg, node, feasibility):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
        self.node = node
        self.cfg = cfg
        if feasibility == 'probabilistic':
            self.feasible_function = self.probabilistic_feasibility
        elif feasibility == 'deterministic':
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
    def __init__(self, dataset_X, dataset_Y, cfg, node, feasibility):
        super().__init__(dataset_X, dataset_Y, cfg, node, feasibility)

    def get_feasible(self, return_indices=True):
        return self.feasible_function(self.dataset_X, self.dataset_Y, return_indices)

    def probabilistic_feasibility(self, X, Y, return_indices=True):
        """
        Method to evaluate the probabilistic feasibility of the data

        X : N x n_d + n_u matrix
        Y : N x n_p x n_g matrix

        """
        n_s = Y.shape[1]

        if self.cfg.samplers.notion_of_feasibility:
            y = jnp.min(Y, axis=-1).reshape(Y.shape[0],Y.shape[1])
            indicator = jnp.where(y>=0, 1, 0)
        else:
            y = jnp.max(Y, axis=-1).reshape(Y.shape[0],Y.shape[1])
            indicator = jnp.where(y<=0, 1, 0)

        prob_feasible = jnp.sum(indicator, axis=1)/n_s

        if not return_indices:
            return X, prob_feasible
        else:
            return X, prob_feasible, (prob_feasible >= self.cfg.samplers.unit_wise_target_reliability[self.node]).squeeze()

    def deterministic_feasibility(self, X, Y, return_indices=True):
        """
        Method to evaluate the deterministic feasibility of the data
        """
        if self.cfg.samplers.notion_of_feasibility == 'positive':
            select_cond = jnp.max(Y, axis=-1)  >= 0 
        else:
            select_cond = jnp.max(Y, axis=-1)  <= 0  
        if not return_indices:
            return X, Y
        else:
            return  X, Y, select_cond.squeeze()
