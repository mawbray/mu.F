from casadi import MX, nlpsol, Function 
import numpy as np
import time
import jax.numpy as jnp
from jax import jit, lax, jacfwd
from jaxopt import LBFGSB
from functools import partial
import time

from constraints.solvers.utilities import (
    build_constraint_functions, build_objective_function, casadify_constraints, unpack_problem_data, unpack_results, clean_up
)

"""
utilities for Casadi NLP solver with general constraints
"""

def casadi_nlp_optimizer_no_gcons(objective, bounds, initial_guess):
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]

    x = MX.sym('x', n_d,1)
    j = objective(x)
    F = Function('F', [x], [j])

    lbx = lb
    ubx = ub
    nlp = {'x':x , 'f':F(x)}

    options = {"ipopt": {"hessian_approximation": "limited-memory"}, 'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 150} 
    solver = nlpsol('solver', 'ipopt', nlp, options)

    solution = solver(x0=np.hstack(initial_guess), lbx=lbx, ubx=ubx)
      
    del nlp, F, x, j, lbx, ubx, options
      
    return solver, solution

def casadi_nlp_optimizer_gcons(objective, constraints, bounds, initial_guess, lhs, rhs):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    Operates in a session via the casadi callbacks and tensorflow V1
    """
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]

    # Get the casadi callbacks required 
    # casadi work up
    x = MX.sym('x', n_d,1)
    j = objective(x)
    g = constraints(x)

    F = Function('F', [x], [j])
    G = Function('G', [x], [g])

    # Define the box bounds
    lbx = lb
    ubx = ub

    # Define the bounds for the equality constraints
    lbg = np.array(lhs)
    ubg = np.array(rhs)

    # Define the NLP
    nlp = {'x':x , 'f':F(x), 'g': G(x)}

    # Define the IPOPT solver
    options = {"ipopt": {"hessian_approximation": "limited-memory"}, 'ipopt.print_level':1, 'print_time':0, 'ipopt.max_iter': 150} 
    solver = nlpsol('solver', 'ipopt', nlp, options)

    # Solve the NLP
    solution = solver(x0=np.hstack(initial_guess), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    del nlp, F, G, x, j, g, lbx, ubx, lbg, ubg, options

    return solver, solution


def casadi_multi_start(initial_guess, objective_func, constraints, bounds):
    n_starts = initial_guess.shape[0]
    n_g = constraints(initial_guess[0].squeeze()).size
    solutions = []
    for i in range(n_starts):
        solver, solution = casadi_nlp_optimizer_gcons(
            objective_func, constraints, bounds, np.array(initial_guess[i,:]).squeeze(),
            lhs=0, rhs=0, n_g=n_g
        )
        if solver.stats()['success']:
          solutions.append((solver, solution))
          if np.array(solution['f']) <= 0: break

    try:
        min_obj_idx = np.argmin(np.vstack([sol_f[1]['f'] for sol_f in solutions]))
        solver_opt, solution_opt = solutions[min_obj_idx]
        n_s = len(solutions)
        del solutions
        return solver_opt, solution_opt, n_s
    except: 
        return None, None, len(solutions)


def ray_casadi_multi_start(problem_id, problem_data, cfg):
  """
  objective: casadi callback
  equality_constraints: casadi callback
  bounds: list
  initial_guess: numpy array
  """
  # TODO update this to handle the case where the problem_data is a dictionary and the contraints are inequality constraints
  initial_guess, bounds, lhs, rhs, n_d, n_starts = unpack_problem_data(problem_data)
  
  # build problem functions
  g_fn = build_constraint_functions(cfg, problem_data)
  objective_fn = build_objective_function(cfg, problem_data, n_d)

  # determine if there are any constraints
  if len(g_fn) > 0:
    casadify_constraints_fn = casadify_constraints(g_fn, initial_guess[0].reshape(1,-1), n_d)
    optimizer_func = partial(casadi_nlp_optimizer_gcons, constraints=casadify_constraints_fn, lhs=lhs, rhs=rhs)
  else:
    casadify_constraints_fn = None
    optimizer_func = casadi_nlp_optimizer_no_gcons

  # run multi start and store solutions
  solutions = []
  for i in range(n_starts):
      solver, solution = optimizer_func(objective=objective_fn, bounds=bounds, initial_guess=np.array(initial_guess[i,:]).squeeze())
      if solver.stats()['success']:
        solutions.append((solver, solution))
        if np.array(solution['f']) <= 0: break

  # unpack and clean up
  solver, solution, ns = unpack_results(solutions, solver, solution)
  clean_up([objective_fn, g_fn, casadify_constraints_fn])
  return solver, solution, ns
  
"""
utilities for JaxOpt box-constrained NLP solver
"""

def multi_start_solve_bounds_nonlinear_program(initial_guess, objective_func, bounds_, tol=1e-4):
    """
    objective is a partial function which just takes as input the decision variables and returns the objective value
    constraints is a vector valued partial function which just take as input the decision variables and return the constraint value, in the form np.inf \leq g(x) \leq 0
    bounds is a list with 
    """
    solutions = []


    partial_jax_solver = jit(partial(solve_nonlinear_program_bounds_jax_uncons, objective_func=objective_func, bounds_=bounds_, tol=tol))

    # iterate over upper level initial guesses
    time_now  = time.time()
    _, solutions = lax.scan(partial_jax_solver, init=None, xs=(initial_guess))
    now = time.time() - time_now

    # iterate over solutions from one of the upper level initial guesses
    assess_subproblem_solution = partial(return_most_feasible_penalty_subproblem_uncons, objective_func=objective_func)
    _, assessment = lax.scan(assess_subproblem_solution, init=None, xs=solutions.params)

    cond = solutions[1].error <= jnp.array([tol]).squeeze()
    mask = jnp.asarray(cond)
    update_assessment = (jnp.where(mask, assessment[0], jnp.minimum(assessment[0],jnp.linalg.norm(assessment[1], axis=1).squeeze())), jnp.where(mask, jnp.linalg.norm(assessment[1], axis=1).squeeze(), jnp.inf))

    # assessment of solutions
    arg_min = jnp.argmin(update_assessment[0], axis=0) # take the minimum objective val
    min_obj = update_assessment[0][arg_min]  # take the corresponding objective value
    min_grad = update_assessment[1][arg_min]# take the corresponding l2 norm of objective gradient

    

    return min_obj.squeeze(), solutions[1].error[arg_min].squeeze()


def solve_nonlinear_program_bounds_jax_uncons(init, xs, objective_func, bounds_, tol):
    """
    objective is a partial function which just takes as input the decision variables and returns the objective value
    bounds is a list 
    # NOTE here we can use jaxopt.ScipyBoundedMinimize to handle general box constraints in JAX.
    # define a partial function by setting the objective and constraint functions and bounds
    # carry init is a tuple (index, list of problem solutions) latter is updated at each iteration
    # x defines a pytree of mu and ftol values
    """
    (x0) = xs

    # Define the optimization problem
    lbfgsb = LBFGSB(fun=objective_func, maxiter=200, use_gamma=True, verbose=False, linesearch="backtracking", decrease_factor=0.8, maxls=100, tol=tol)

    problem = lbfgsb.run(x0, bounds=bounds_) # 


    return None, problem

def return_most_feasible_penalty_subproblem_uncons(init, xs, objective_func):
    # iterate over upper level initial guesses
    solution = xs
    
    # get gradients of solutions, value of objective and value of constraints
    return None, (objective_func(solution), jacfwd(objective_func)(solution))



