
import numpy as np
from abc import ABC


class live_set:
    def __init__(self, cfg, notion_of_feasibility):
        self.cfg = cfg
        self.live_set, self.live_set_prob = [], []
        self.notion_of_feasibility = notion_of_feasibility
        self.infeasible_set, self.infeasible_prob = [], []
        self.acceptanceratio = 0
        self.N = 0
    
    def acceptance_ratio(self, feasible):

        average_ = np.sum(feasible)
        current = self.acceptanceratio
        update = average_ + current * self.N
        update /= (feasible.shape[0] + self.N)
        self.acceptanceratio = update
        return 
        

    def evaluate_feasibility(self, x):
        """
        Evaluate the feasibility of the live set
        :param x: The input
        :return: The feasibility
        """
        n_s = x.shape[1]

        if self.notion_of_feasibility:
            y = np.min(x, axis=-1).reshape(x.shape[0],x.shape[1])
            indicator = np.where(y>=0, 1, 0)
            prob_feasible = np.sum(indicator, axis=1)/n_s
            return prob_feasible >= self.cfg.samplers.target_reliability, prob_feasible
        else:
            y = np.max(x, axis=-1).reshape(x.shape[0],x.shape[1])
            indicator = np.where(y<=0, 1, 0)
            prob_feasible = np.sum(indicator, axis=1)/n_s
            return prob_feasible >= self.cfg.samplers.target_reliability, prob_feasible
             
        
    def check_live_set_membership(self, x, g):
        """
        Check the membership of the live set
        :param x: The input
        :return: The membership
        """
        # determine those points that are feasible
        feasible, prob = self.evaluate_feasibility(g)
        feasible_points = x[feasible, :]
        feasible_prob = prob[feasible]
        # store infeasible points within a data holder (should probably be a separate class)
        self.infeasible_set.append(x[~feasible, :])
        self.infeasible_prob.append(prob[~feasible].reshape(-1,1))
        # update acceptance ratio 
        self.acceptance_ratio(feasible=feasible)
        return feasible_points, feasible_prob
    
    def append_to_live_set(self, x, y):
        """
        Append to the live set
        :param x: The input
        """
        self.live_set.append(x)
        self.live_set_prob.append(y.reshape(-1,1))
        return
    
    def get_live_set(self):
        """
        Get the live set
        :return: The live set
        """
        return np.vstack(self.live_set)[:self.live_set_len(), :], np.vstack(self.live_set_prob)[:self.live_set_len()]
    
    def live_set_len(self):
        """
        Get the length of the live set
        :return: The length of the live set
        """
        return min(np.vstack(self.live_set).shape[0], self.cfg.samplers.ns.final_sample_live)

    def check_if_live_set_complete(self):
        if np.vstack(self.live_set).shape[0] >= self.cfg.samplers.ns.final_sample_live:
            print(np.vstack(self.live_set).shape, self.cfg.samplers.ns.final_sample_live)
            return True
        else:
            return False
        
    def load_classification_data_to_graph(self, graph=None, str='classifier_training'):
        """    Get the classification data for training a classifier
        :param graph: The graph
        :param cfg: The configuration
        :return: The support and labels for the classifier
        """
        if graph is None:
            raise ValueError("Graph must be provided to load classification data.")

        # get samples
        live_set = np.vstack(self.live_set)
        infeasible_set = np.vstack(self.infeasible_set)
        # corresponding labels
        live_set_labels = np.vstack(self.live_set_prob)
        infeasible_set_labels = np.vstack(self.infeasible_prob)
        # create a dataset object
        all_data = np.vstack([live_set, infeasible_set])    
        all_labels = np.vstack([live_set_labels, infeasible_set_labels])
        graph.graph[str] = dataset(all_data, all_labels)
        return graph
    


class dataset(ABC):
    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else np.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else np.expand_dims(y, axis=-1)
            
        