
import numpy as np



class live_set:
    def __init__(self, cfg, notion_of_feasibility):
        self.cfg = cfg
        self.live_set = []
        self.notion_of_feasibility = notion_of_feasibility
    
    def evaluate_feasibility(self, x):
        """
        Evaluate the feasibility of the live set
        :param x: The input
        :return: The feasibility
        """
        if self.notion_of_feasibility:
            return np.min(x, axis=1).squeeze() >= 0
        else:
            return np.max(x, axis=1).squeeze() <= 0

    def check_live_set_membership(self, x, g):
        """
        Check the membership of the live set
        :param x: The input
        :return: The membership
        """
        feasible = self.evaluate_feasibility(np.vstack(g))
        feasible_points = x[feasible, :]
        return feasible_points
    
    def append_to_live_set(self, x):
        """
        Append to the live set
        :param x: The input
        """
        self.live_set.append(x)
        return
    
    def get_live_set(self):
        """
        Get the live set
        :return: The live set
        """
        return np.vstack(self.live_set)[:self.live_set_len(), :]
    
    def live_set_len(self):
        """
        Get the length of the live set
        :return: The length of the live set
        """
        return min(np.vstack(self.live_set).shape[0], self.cfg.ns.final_sample_live)

    def check_if_live_set_complete(self):
        if np.vstack(self.live_set).shape[0] >= self.cfg.ns.final_sample_live:
            print(np.vstack(self.live_set).shape, self.cfg.ns.final_sample_live)
            return True
        else:
            return False