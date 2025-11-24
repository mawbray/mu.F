from abc import ABC
from functools import partial
import jax.numpy as jnp
import logging
import ray
import numpy as np

from constraints.solvers.functions import generate_initial_guess, multi_start_solve_bounds_nonlinear_program, ray_casadi_multi_start

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
    


class serialms_casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, constraints, bounds):
        super().__init__(cfg)
        self.solver, self.problem_data = self.construct_solver(objective_func, constraints, bounds)

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

    def construct_solver(self, objective_func, constraints, bounds):
        # formatting for casadi
        self.n_d = len(bounds[0])
        self.bounds = bounds
        solver = ray_casadi_multi_start
        problem_data = {'objective_func': objective_func, 'constraints': constraints, 'bounds': bounds}
        problem = {'data': problem_data}
        return solver, problem
    
    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def get_message(self, solver):
        return solver['return_status']
    
    def solve_digest(self, solver, result, len_feasible):
        message = self.get_message(solver)
        status = self.get_status(solver)
        objective = self.get_objective(result)
        constraints = self.get_constraints(result)
        t_wall = self.get_time(solver)
        decision = self.get_solution(result)

        if not status:
            objective = np.maximum(np.array([objective]).reshape(-1,), np.max(np.absolute(constraints)).reshape(-1,))

        if (t_wall >= self.cfg.max_solution_time) and (not status):
            logging.warning(f'--- Forward solver max time exceeded: {t_wall} s ---')
        
        if (not status):
            logging.warning(f'{message}')

        del solver, result, t_wall, len_feasible

        return {'success': status, 'objective': objective, 'constraints': constraints, 'message': message, 'decision_variables': decision}
    
    def get_solution(self, solution):
        return solution['x']

    def get_status(self, solver):
        try:
            return solver['success']
        except:
            return False
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['g']
    
    def get_time(self, solver):
        return sum([x for k,x in solver.items() if 't_wall_' in k])

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
        solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=self.objective_func, bounds_=(self.bounds[0], self.bounds[1]), tol=self.cfg.jax_opt_options.error_tol)
        objective, error = solver(initial_guesses)

        del solver

        return {'objective': objective, 'error': error}
    
    def get_status(self, stationary_error):
        return jnp.norm(stationary_error) <= self.cfg.jax_opt_options.error_tol 

    def get_objective(self, objective):
        return objective
    

class parallel_casadi_box_eq_nlp_solver(solver_base):
    def __init__(self, cfg, objective_func, constraints, bounds):
        super().__init__(cfg)
        self.solver, self.problem_data = self.construct_solver(objective_func, constraints, bounds)

    def __call__(self, initial_guesses):
        return self.solve(initial_guesses)

    def construct_solver(self, objective_func, constraints, bounds):
        # formatting for casadi
        self.n_d = len(bounds[0])
        self.bounds = bounds
        solver = ray.remote(ray_casadi_multi_start) # , num_cpus=1)
        problem_data = {'objective_func': objective_func, 'constraints': constraints, 'bounds': bounds}
        problem = {'data': problem_data}
        return solver, problem
    
    def initial_guess(self):
        return generate_initial_guess(self.cfg.n_starts, self.n_d, self.bounds)
    
    def get_message(self, solver):
        return solver['return_status']
    
    def solve_digest(self, solver, result, len_feasible):
        message = self.get_message(solver)
        status = self.get_status(solver)
        objective = self.get_objective(result)
        constraints = self.get_constraints(result)
        t_wall = self.get_time(solver)
        decision = self.get_solution(result)

        if not status:
            objective = np.maximum(np.array([objective]).reshape(-1,), np.max(np.absolute(constraints)).reshape(-1,))

        if (t_wall >= self.cfg.max_solution_time) and (not status):
            logging.warning(f'--- Forward solver max time exceeded: {t_wall} s ---')
        
        if (not status):
            logging.warning(f'{message}')

        del solver, result, t_wall, len_feasible

        return {'success': status, 'objective': objective, 'constraints': constraints, 'message': message, 'decision_variables': decision}
    
    
    def get_status(self, solver):
        try:
            return solver['success']
        except:
            return False
    
    def get_objective(self, solution):
        return solution['f']
    
    def get_constraints(self, solution):
        return solution['g']
    
    def get_time(self, solver):
        return sum([x for k,x in solver.items() if 't_wall_' in k])
    
    def get_solution(self, solution):
        return solution['x']
    
