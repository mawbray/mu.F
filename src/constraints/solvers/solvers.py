from abc import ABC
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, pmap
import logging
import ray

from constraints.solvers.functions import generate_initial_guess, nlp_multi_start_casadi_eq_cons, multi_start_solve_bounds_nonlinear_program, casadi_nlp_construction,  evaluate_casadi_nlp_ms

class solver_base(ABC):
    """ This is a base class used to construct local solvers for the feasibility problem """
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, initial_guesses):
        raise NotImplementedError
        return self.solve(initial_guesses)
    
    def construct_solver(self):
        raise NotImplementedError
    
    def solve(self):
        raise NotImplementedError
    
    def get_status(self):
        raise NotImplementedError
       
    def get_objective(self):
        raise NotImplementedError
    
    def initial_guess(self):
        raise NotImplementedError
    



class casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, equality_constraints, bounds):
        super().__init__(cfg)
        self.construct_solver(objective_func, equality_constraints, bounds)
    
    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

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
            logging.info('--- Solver did not converge ---')
            logging.info(f'Objective: {objective}')
            logging.info(f'Constraints: {constraints}')
            logging.info(f'Time: wall - {time[0]}, process - {time[1]}')

        return {'success': status, 'time': time, 'objective': objective, 'constraints': constraints}
    
    
    def get_status(self, solver):
        return solver.stats()['success']
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['np']['g']
"""    
@ray.remote
class parallel_casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, equality_constraints, bounds):
        super().__init__(cfg)
        self.construct_solver(objective_func, equality_constraints, bounds)

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

    
    def construct_solver(self, objective_func, equality_constraints, bounds):
        self.n_d = len(bounds[0])
        self.bounds = bounds
        # formatting for casadi
        self.solver_object, self.constraints = casadi_nlp_construction(objective_func=objective_func, equality_constraints=equality_constraints, bounds=bounds)
        self.solver = partial(evaluate_casadi_nlp, solver=self.solver_object, constraints=self.constraints, n_d=self.n_d)
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
            logging.info('--- Solver did not converge ---')
            logging.info(f'Objective: {objective}')
            logging.info(f'Constraints: {constraints}')
            logging.info(f'Time: wall - {time[0]}, process - {time[1]}')

        return {'success': status, 'time': time, 'objective': objective, 'constraints': constraints}
    
    
    def get_status(self, solver):
        return solver.stats()['success']
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['np']['g']
"""    


class parallelms_casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, equality_constraints, bounds):
        super().__init__(cfg)
        self.construct_solver(objective_func, equality_constraints, bounds)

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

    def construct_solver(self, objective_func, equality_constraints, bounds):
        self.n_d = len(bounds[0])
        self.bounds = bounds
        # formatting for casadi
        self.solver = partial(evaluate_casadi_nlp_ms, objective_func=objective_func, equality_constraints=equality_constraints, bounds=bounds, device_count=self.cfg.device_count)
        return
    
    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def solve(self, initial_guesses):
        obj_fn, dec_x = self.solver(initial_guesses)
        status = self.get_status(obj_fn)

        if not status:
            logging.info('--- Solver did not converge ---')

        return {'success': status, 'objective': obj_fn, 'x': dec_x}
    
    def get_status(self, solver):
        return solver < jnp.inf
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['np']['g']


class jax_box_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, bounds):
        super().__init__(cfg)
        self.construct_solver(objective_func, bounds)

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

    def construct_solver(self, objective_func, bounds):
        self.n_d = len(bounds[0])
        self.bounds = bounds
        self.objective_func = objective_func
        self.bounds = bounds
        return    

    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def solve(self, initial_guesses):
        solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=self.objective_func, bounds_=(self.bounds[0], self.bounds[1]))
        objective, objective_grad, error = solver(initial_guesses)
        return {'objective': objective, 'ojective_grad': objective_grad, 'error': error}
    
    def get_status(self, error):
        return error <= self.cfg.jax_opt_options.error_tol 

    def get_objective(self, objective):
        return objective
    