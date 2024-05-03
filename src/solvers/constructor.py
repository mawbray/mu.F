from abc import ABC
from solvers.solvers import casadi_box_eq_nlp_solver, jax_box_nlp_solver


class solver_construction(ABC):
    """ This is a base class used to construct local solvers for the feasibility problem """
    def __init__(self, cfg, solver_type):
        self.cfg = cfg
        self.solver_type = solver_type

    @staticmethod
    def from_method(cfg, solver_type, objective_func, bounds, eq_constraints_func=None):
        """
        Important - This is to be called to actually load the objective function, constraints, and bounds into the solver and return a solver object specific to the subproblem
        - NOTE to users; if anyone finds this particularly problematic, please raise an issue and I will come back to it.    
        """
        new_solver_class = solver_construction(cfg, solver_type)
        if type(objective_func) == type(None):
            raise ValueError('Objective function must be provided')
        if type(eq_constraints_func) == type(None):
            raise Warning('No equality constraints provided; assuming that we will solve a box-constrained problem')
        else: new_solver_class.load_equality_constraints(eq_constraints_func)
        new_solver_class.load_objective(objective_func)
        new_solver_class.load_bounds(bounds)
        new_solver_class.construct_solver()
        return new_solver_class

    def __call__(self, initial_guesses):
        return self.solver.solve(initial_guesses)
    
    def construct_solver(self):
        if self.solver_type == 'general_constrained_nlp':
            self.solver = casadi_box_eq_nlp_solver(self.cfg, self.objective_func, self.constraints_func, self.bounds)
        elif self.solver_type == 'box_constrained_nlp':
            self.solver = jax_box_nlp_solver(self.cfg, self.objective_func, self.bounds)
        else: 
            raise NotImplementedError(f'Solver type {self.solver_type} not implemented')
        
    def load_objective(self, objective_func):
        self.objective_func = objective_func
    
    def load_equality_constraints(self, constraints_func):
        self.constraints_func = constraints_func
    
    def load_bounds(self, bounds):
        self.bounds = bounds
        
    def initial_guess(self):
        return self.solver.initial_guess()