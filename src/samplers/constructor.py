from abc import ABC 
from typing import Type
import pickle
import logging
import time

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
        self.sampler_class(self.problem_description)
        return

    def load_model_to_sampler(self):
        raise NotImplementedError("load_model_to_sampler should be implemented in the derived class")

    def solve(self):
        t0 = time.time()
        self.sampler_class.solve()
        logging.info(f"Time taken to solve the problem: {time.time() - t0} seconds")
        return 

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
        self.problem_description['solver']['score_evaluation']['constraints_func_ptr'] = self.model.get_constraints
        self.problem_description['solver']['efp_evaluation']['constraints_func_ptr'] = self.model.get_constraints
        return

    def get_solution(self):
        pd = self.problem_description
        output = load_study(pd)
        feasible_samples, infeasible_samples = return_solution(pd, output)
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
        outside_samples_coords = np.empty((0, n_d))
        for i, phi in enumerate(samples["phi"]):
            if phi >= problem_description["problem"]["target_reliability"]:
                inside_samples_coords = np.append(inside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
            else:
                outside_samples_coords = np.append(outside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
                
        return inside_samples_coords, outside_samples_coords


    