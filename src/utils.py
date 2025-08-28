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
            G.edges[predec, node]["aux_filter"] = None # drop backward surrogates
        G.nodes[node]["constraints"] = None   # drop constraint functions (all of these are non-serializable.)
        G.nodes[node]["constraints_vmap"] = None # drop backward evaluators
        G.nodes[node]['classifier'] = None 
        G.nodes[node]['q_function'] = None
        G.nodes[node]['unit_params_fn'] = None


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
    def __init__(self, dataset_object, index_on=None):
        self.dataset_object = dataset_object
        self.index_on = index_on
        if index_on is None:
            self.d = dataset_object.d
            self.p = dataset_object.p
            self.y = dataset_object.y
        else: 
            self.d = dataset_object.d[index_on:]
            self.p = dataset_object.p[index_on:]
            self.y = dataset_object.y[index_on:]
        
        self.check_data()

    def check_data(self):
        n_d = len(self.d)
        assert n_d == len(self.p) == len(self.y), "The number of design, parameters and outputs objects within the data object should be the same, currently {} n_d, {} n_p and {} n_o".format(n_d, len(self.p), len(self.y))

    def get_data(self):
        for i in range(len(self.d)):
            yield self.d[i], self.p[i], self.y[i]

    def transform_data_to_matrix(self, edge_fn, feasible_indices=None, index_on=None):
        """
        Dealing with the data in a matrix format
         - This is useful for training neural networks / any other input-output model
        """
        data = self.get_data()
        data_store_X, data_store_Y, data_store_Z = [], [], []
        if feasible_indices is not None: data = zip(data, feasible_indices)
        else: data = zip(data)
        for index, data_zip in enumerate(data): 
            # skip the data collected in sampling iterations until index_on
            if index_on is not None:
                if index<index_on:
                    pass
            # if feasible_indices is not None, then we have feasible indices to unpack
            if feasible_indices is not None:
                (d, p, y), fe_ind = data_zip
            else:
                d, p, y = data_zip[0]
            # preparation
            X, Y, Z = [], [], []
            if p.ndim<2: p=p.reshape(1,-1)
            if d.ndim<2: d=d.reshape(1,-1)
            if p.ndim < 3: p = jnp.expand_dims(p, axis=1)
            if d.ndim < 3: d = jnp.expand_dims(d, axis=1)

            y_edge = edge_fn(y)
            if y_edge.ndim < 3: y_edge = jnp.expand_dims(y_edge, axis=-1)

            # loop over the number of uncertain parameters to create input-output data (handling vectorization of the model evaluations)
            for i in range(p.shape[0]):
                X.append(jnp.hstack([d.reshape((d.shape[0], d.shape[1])), jnp.repeat(p[i].reshape(1,-1),d.shape[0], axis=0)]))
                Y.append(y_edge[:,i,:].reshape(d.shape[0],-1))
                if feasible_indices is not None:
                    # This is useful for updating KS or fitting models to feasible region only.
                    Z.append(y_edge[:,i,:].reshape(d.shape[0],-1)[fe_ind.squeeze()])

            X = jnp.vstack(X)
            Y = jnp.vstack(Y)
            Z = jnp.vstack(Z) if feasible_indices is not None else None
            data_store_X.append(X)
            data_store_Y.append(Y)
            data_store_Z.append(Z)

        if feasible_indices is not None:
            return jnp.vstack(data_store_X), jnp.vstack(data_store_Y), jnp.vstack(data_store_Z)
        else:
            return jnp.vstack(data_store_X), jnp.vstack(data_store_Y), None
    

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

    def probabilistic_feasibility(self, return_indices=True):
        """
        Method to evaluate the probabilistic feasibility of the data

        X : N x n_d + n_u matrix
        Y : N x n_p x n_g matrix

        """
        X, Y = self.dataset_X, self.dataset_Y

        Y_s = []
        for x, y in zip(X, Y):

            if self.cfg.samplers.notion_of_feasibility == 'positive':
                y = jnp.min(y, axis=-1).reshape(y.shape[0],y.shape[1])
                indicator = jnp.where(y>=0, 1, 0)
            else:
                y = jnp.max(y, axis=-1).reshape(y.shape[0],y.shape[1])
                indicator = jnp.where(y<=0, 1, 0)

            n_s = y.shape[1]
            prob_feasible = jnp.sum(indicator, axis=1)/n_s
            Y_s.append(prob_feasible.reshape(y.shape[0],1))

        if not return_indices:
            return jnp.vstack(X), jnp.vstack(Y_s)
        else:
            return jnp.vstack(X), jnp.vstack(Y_s), [ys >= self.cfg.samplers.unit_wise_target_reliability[self.node] for ys in Y_s]

    def deterministic_feasibility(self, X, Y, return_indices=True):
        """
        Method to evaluate the deterministic feasibility of the data
        """
        cond = []
        labels = []
        for x, y in zip(X,Y):
            if self.cfg.samplers.notion_of_feasibility == 'positive':
                lab = jnp.min(y, axis=-1)
                select_cond = lab  >= 0 
            else:
                lab = jnp.max(y, axis=-1) 
                select_cond = lab  <= 0  

            cond.append(select_cond)
            labels.append(lab)
            

        if not return_indices:
            return jnp.vstack(X), jnp.vstack(labels)
        else:
            return  jnp.vstack(X), jnp.vstack(labels), cond
        


       
