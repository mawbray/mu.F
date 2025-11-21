from abc import ABC 
from typing import Type
import pickle
import logging
import time
import numpy as np

class construct_base(ABC):
    def __init__(self, sampler_class, problem_description: dict, model):
        """ 
        Sampler is a sampling-based method that is used to solve the problem.
        The problem description is a dictionary that contains the problem description.
        """

        self.problem_description = problem_description
        self.model = model
        self.sampler_class =  sampler_class
        self.load_model_to_sampler()
        self.construct_problem()

    def construct_problem(self):
        self.sampler = self.sampler_class(self.problem_description)
        return

    def load_model_to_sampler(self):
        raise NotImplementedError("load_model_to_sampler should be implemented in the derived class")

    def solve(self):
        """ This is up to you to implement and is solver dependent """
        raise NotImplementedError("solve should be implemented in the derived class")

    def get_solution(self):
        """ This is up to you to implement and is solver dependent """
        raise NotImplementedError("get_solution should be implemented in the derived class")

    def get_function_evaluations(self):
        """ This is up to you to implement and is solver dependent """
        raise NotImplementedError("get_function_evaluations should be implemented in the derived class")


class construct_deus_problem(construct_base):
    def __init__(self, sampler_class, problem_description: dict, model):
        super().__init__(sampler_class, problem_description, model)

    def load_model_to_sampler(self):
        self.problem_description['solver']['settings']['score_evaluation']['constraints_func_ptr'] = self.model.get_constraints
        self.problem_description['solver']['settings']['efp_evaluation']['constraints_func_ptr'] = self.model.get_constraints
        return
    
    def solve(self):
        t0 = time.time()
        self.sampler.solve()
        logging.info(f"Time taken to solve the problem: {time.time() - t0} seconds")
        return 
    
    def get_log_evidence(self):
        pd = self.problem_description
        output = self.load_study(pd)
        return output["solution"]["log_z"]

    def get_solution(self):
        pd = self.problem_description
        output = self.load_study(pd)
        feasible_samples, infeasible_samples = self.return_solution(pd, output)
        return feasible_samples, infeasible_samples

    @staticmethod
    def load_study(problem_description):
        cs_path = problem_description["activity_settings"]["case_path"]
        cs_name = problem_description["activity_settings"]["case_name"]

        with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') \
                as file:
            output = pickle.load(file)

        return output

    @staticmethod
    def return_solution(problem_description, output):
        samples = output["solution"]["probabilistic_phase"]["samples"]
        if samples["coordinates"] == []:
            samples = output["solution"]["deterministic_phase"]["samples"]
        n_d = len(samples["coordinates"][0])
        inside_samples_coords = np.empty((0, n_d))
        inside_samples_prob = []
        outside_samples_coords = np.empty((0, n_d))
        outside_sample_prob = []
        for i, phi in enumerate(samples["phi"]):
            if phi >= problem_description["problem"]["target_reliability"]:
                inside_samples_coords = np.append(inside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
                inside_samples_prob.append(samples["phi"][i])
            else:
                outside_samples_coords = np.append(outside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
                outside_sample_prob.append(samples["phi"][i])

        if inside_samples_coords.shape[0] == 0:
            logging.warning("Warning!! ---- No feasible samples found")
                
        return (inside_samples_coords, np.array(inside_samples_prob)), (outside_samples_coords, np.array(outside_sample_prob))


    
class construct_deus_problem_network(construct_deus_problem):
    def __init__(self, sampler_class, problem_description: dict, model):
        super().__init__(sampler_class, problem_description, model)

    def load_model_to_sampler(self):
        self.problem_description['solver']['settings']['score_evaluation']['constraints_func_ptr'] = self.model.direct_evaluate
        self.problem_description['solver']['settings']['efp_evaluation']['constraints_func_ptr'] = self.model.direct_evaluate
        return
    

    