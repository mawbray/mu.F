from abc import ABC
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, pmap

from functions import generate_initial_guess, nlp_multi_start_casadi_eq_cons, multi_start_solve_bounds_nonlinear_program

class solver_base(ABC):
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
    

class casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, logging, cfg, objective_func, equality_constraints, bounds):
        super().__init__(logging, cfg)
        self.construct_solver(objective_func, equality_constraints, bounds)
    
    def construct_solver(self, objective_func, equality_constraints, bounds):
        self.n_d = len(bounds[0])
        self.bounds = bounds
        self.solver = partial(nlp_multi_start_casadi_eq_cons, objective_func=objective_func, equality_constraints=equality_constraints, bounds=bounds)
        return
    
    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def solve(self, initial_guesses):
        solver, solution = self.solver(initial_guesses)
        status = self.get_status(solver)
        time = self.get_time(solver)
        objective = self.get_objective(solution)
        constraints = self.get_constraints(solution)

        if not status:
            self.logging.info('--- Solver did not converge ---')
            self.logging.info(f'Objective: {objective}')
            self.logging.info(f'Constraints: {constraints}')
            self.logging.info(f'Time: wall - {time[0]}, process - {time[1]}')

        return status, time, objective, constraints
    
    
    def get_status(self, solver):
        return solver.stats()['success']
    
    def get_time(self, solver):
        return solver.stats()['t_wall_total'], solver.stats()['t_proc']
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['np']['g']
    


class jax_box_nlp_solver(solver_base):
    def __init__(self, logging, cfg, objective_func, bounds):
        super().__init__(logging, cfg)
        self.construct_solver(objective_func, bounds)

    def construct_solver(self, objective_func, bounds):
        self.n_d = len(bounds[0])
        self.bounds = bounds
        self.solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=objective_func, bounds=(bounds[0], bounds[1]))
        return    

    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def solve(self, initial_guesses):
        objective, objective_grad, error = self.solver(initial_guesses)
        
        return objective, objective_grad, error
    
    def get_status(self, error):
        return error <= self.cfg.box_constrained.tol 

    def get_objective(self, objective):
        return objective
    