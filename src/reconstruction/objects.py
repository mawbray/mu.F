
import numpy as np



class live_set:
    def __init__(self, cfg, notion_of_feasibility):
        self.cfg = cfg
        self.live_set, self.live_set_prob = [], []
        self.notion_of_feasibility = notion_of_feasibility
        self.dead_set, self.dead_set_prob = [], []
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
        feasible, prob = self.evaluate_feasibility(g)
        feasible_points = x[feasible, :]
        feasible_prob = prob[feasible]
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