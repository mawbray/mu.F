from abc import ABC

from solvers import casadi_box_eq_nlp_solver, jax_box_nlp_solver


class solver_construct_base(ABC):
    """ This is a base class used to construct local solvers for the feasibility problem """
    def __init__(self, logging, cfg):
        self.logging = logging
        self.cfg = cfg

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)
    
    def construct_solver(self):
        raise NotImplementedError
    
    def solve(self):
        raise NotImplementedError
    
    def get_status(self):
        raise NotImplementedError
    
    def get_time(self):
        raise NotImplementedError
    
    def get_objective(self):
        raise NotImplementedError
    
    def get_constraints(self):
        raise NotImplementedError
    
    def initial_guess(self):
        raise NotImplementedError