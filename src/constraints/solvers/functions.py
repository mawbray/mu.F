from casadi import DM, MX, nlpsol, Function, Sparsity, Callback
import casadi 
import numpy as np
from scipy.stats import qmc
import time
import jax.numpy as jnp
from jax import jacobian, jit, vmap, lax, jacfwd
from jaxopt import LBFGSB
from functools import partial
import time

from constraints.solvers.surrogate.surrogate import surrogate_reconstruction

"""
utilities for Casadi NLP solver with equality constraints
"""

def casadi_nlp_optimizer_no_gcons(objective, bounds, initial_guess):
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]
    
    cost_fn   = casadify(objective, n_d, output_dim=1)

    x = MX.sym('x', n_d,1)
    j = cost_fn(x)
    F = Function('F', [x], [j])

    lbx = lb
    ubx = ub
    nlp = {'x':x , 'f':F(x)}

    options = {"ipopt": {"hessian_approximation": "limited-memory"}, 'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 150} 
    solver = nlpsol('solver', 'ipopt', nlp, options)

    solution = solver(x0=np.hstack(initial_guess), lbx=lbx, ubx=ubx)
      
    del cost_fn, nlp, F, x, j, lbx, ubx, options
      
    return solver, solution

def casadi_nlp_optimizer_eq_cons(objective, equality_constraints, bounds, initial_guess, lhs, rhs):
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
    cost_fn   = casadify(objective, n_d)
    eq_cons   = casadify(equality_constraints, n_d)

    # casadi work up
    x = MX.sym('x', n_d,1)
    j = cost_fn(x)
    g = eq_cons(x)

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
    
    del cost_fn, eq_cons, nlp, F, G, x, j, g, lbx, ubx, lbg, ubg, options
    
    return solver, solution


def casadi_multi_start(initial_guess, objective_func, constraints, bounds):
    n_starts = initial_guess.shape[0]
    n_g = constraints(initial_guess[0].squeeze()).size
    solutions = []
    for i in range(n_starts):
        solver, solution = casadi_nlp_optimizer_eq_cons(
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
  initial_guess, bounds, lhs, rhs = \
    problem_data['initial_guess'], problem_data['bounds'], problem_data['eq_lhs'], problem_data['eq_rhs']
  n_starts = initial_guess.shape[0]

  # get constraint functions and define masking of the inputs
  g_fn = {}
  for i, cons_data in enumerate(problem_data['constraints'].values()):
    fn = construct_model(cons_data['params'], cfg, 
                          supervised_learner=cons_data['model_class'],
                          model_type=cons_data['model_type'],
                          model_surrogate=cons_data['model_surrogate'])
    if problem_data['uncertain_params'] == None:
      g_fn[i] = partial(cons_data['g_fn'], fn = fn)
    else:
        raise NotImplementedError("Uncertain parameters not yet implemented for inequality constraints")
  
  # define the constraints function
  constraints = partial(lambda x, g: jnp.vstack([g[i](x) for i in range(len(g))]), g=g_fn)

  # get objective function
  obj_data = problem_data['objective_func']
  n_f = len([k for k in list(obj_data.keys()) if 'f' in k])
  if n_f > 1:
    # objective requires some function composition
    obf = construct_model(obj_data['f0']['params'], cfg, supervised_learner=obj_data['f0']['model_class'], model_type=obj_data['f0']['model_type'], model_surrogate=obj_data['f0']['model_surrogate'])
    if n_f > 2:
      obj_terms = {}
      for i in range(1,n_f-1):
        eqc = construct_model(obj_data[f'f{i}']['params'], cfg, supervised_learner=obj_data[f'f{i}']['model_class'], model_type=obj_data[f'f{i}']['model_type'], model_surrogate=obj_data[f'f{i}']['model_surrogate'])
        obj_terms[i-1] = partial(lambda x, v : eqc(x.reshape(1,-1)[:,v].reshape(-1,)).reshape(-1,1), v = jnp.array(obj_data[f'f{i}']['args']))
      # construct objective from constituent functions
      obj_in = partial(lambda x, g: jnp.hstack([g[i](x) for i in range(len(g))]), g=obj_terms)
      objective_func = partial(obj_data['obj_fn'], f1=obf, f2=obj_in)
    else:
      objective_func = partial(obj_data['obj_fn'], f1=obf)
  else:
      print(obj_data)
      objective_func = lambda x: x.reshape(-1,)[obj_data['obj_fn']].reshape(1,1)
    
  # store for solutions
  solutions = []
  for i in range(n_starts):
      if len(g_fn) >0:
        solver, solution = casadi_nlp_optimizer_eq_cons(objective_func, constraints, bounds, np.array(initial_guess[i,:]).squeeze(), lhs, rhs)
      else: 
        solver, solution = casadi_nlp_optimizer_no_gcons(objective_func, bounds, np.array(initial_guess[i,:]).squeeze())
      if solver.stats()['success']:
        solutions.append((solver, solution))
        if np.array(solution['f']) <= 0: break

  try:
      min_obj_idx = np.argmin(np.vstack([sol_f[1]['f'] for sol_f in solutions]))
      solver_opt, solution_opt = solutions[min_obj_idx]
      n_s = len(solutions)
      del solutions
      return solver_opt.stats(), solution_opt, n_s
  except: 
      return solver.stats(), solution, len(solutions)
    
# --- Core JAX/CasADi Integration ---

# NOTE currently callbacks are only set up for scalar valued functions
# i.e. functions that map R^n_x -> R
# Combined with reverse mode AD to get gradients
# This is roughly equivalent to reverse mode AD on vector valued functions
# i.e. functions that map R^n_x -> R^n_g but the implementation is different
# in this case we require (m-1) additional constraint evaluations to get the full Jacobian
# However, if we extend Callbacks to handle vector valued functions
# we can use forward mode AD to get Jacobians
# which may be more efficient in many cases (i.e. when n_x < n_g)


class JaxCasADiEvaluator(Callback):
    def __init__(self, t_in, t_out, model, set_init=False, opts={}):
        self.set_init = set_init
        self.opts = opts
        Callback.__init__(self)
        assert isinstance(t_in, list)
        self.t_in = t_in
        assert isinstance(t_out, list)
        self.t_out = t_out
        self.output_shapes = []
        self.construct("JaxCasADiEvaluator", {})
        self.refs = []
        self.model = model
        self.jitted_model = jit(vmap(self.model))
        self.jitted_grad_func = jit(
            jacobian(lambda x: self.jitted_model(jnp.atleast_2d(x)).squeeze())
        )

    def get_n_in(self):
        return len(self.t_in)

    def get_n_out(self):
        return len(self.t_out)

    def get_sparsity_in(self, i):
        tensor_shape = self.t_in[i].shape
        return Sparsity.dense(tensor_shape[0], tensor_shape[1])

    def get_sparsity_out(self, i):
        if i == 0 and not self.set_init:
            tensor_shape = self.opts["output_dim"]
        elif i == 0 and self.set_init:
            tensor_shape = self.opts["grad_dim"]
        else:
            tensor_shape = self.opts["output_dim"]
        return Sparsity.dense(tensor_shape[0], tensor_shape[1])

    def objective_func(self, x):
        x = jnp.atleast_2d(x)
        mean = vmap(self.model)(x)
        return mean.squeeze()

    def eval(self, arg):
        updated_t = []
        for i, v in enumerate(self.t_in):
            if isinstance(arg[i], (MX, DM)):
                arg_np = np.array(arg[i].full()).reshape(v.shape)
            else:
                arg_np = np.array(arg[i]).reshape(v.shape)
            updated_t.append(jnp.array(arg_np))
        input_data = jnp.atleast_2d(updated_t[0]).reshape(1, -1)

        if len(arg) > 1:  # Gradient calculation
            grad_output = self.jitted_grad_func(input_data)
            selected_set = np.array(grad_output).reshape(
                self.opts["grad_dim"][0], self.opts["grad_dim"][1]
            )
        else:  # Function value
            out_ = self.jitted_model(input_data)
            selected_set = np.array(out_).reshape(
                (self.opts["output_dim"][0], self.opts["output_dim"][1])
            )
            if out_ is None:
                raise ValueError("Output from the model is None.")

        return [selected_set]

    def has_reverse(self, nadj):
        return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        adj_seed = [MX.sym("adj", *self.get_sparsity_out(i).shape) for i in range(self.get_n_out())]
        callback = JaxCasADiEvaluator(
            self.t_in + adj_seed, [self.t_out[0]], self.model, set_init=True, opts=self.opts
        )
        self.refs.append(callback)
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        casadi_bal = callback.call(nominal_in + adj_seed)
        return Function(name, nominal_in + nominal_out + adj_seed, casadi_bal, inames, onames)


class ScalarFn(JaxCasADiEvaluator):
    def __init__(self, model, opts={}):
        X = MX.sym("X", opts["grad_dim"][0], 1)

        @jit
        def f_k(input_dat, get_grad_val=None):
            xf_tensor = jnp.array(input_dat)
            if get_grad_val is not None:
                mean = self.objective_func(xf_tensor)
            # Note: JAX automatically handles gradient computation
            else:
                mean = self.objective_func(xf_tensor)
            return mean, None  # Return None for grad_mean as JAX handles it differently

        JaxCasADiEvaluator.__init__(self, [X], [f_k], model, opts=opts)
        self.counter = 0
        self.time = 0

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret


def casadify(functn, nd, ny=1):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """
    opts = {"output_dim": [1, ny], "grad_dim": [nd, ny]}
    return ScalarFn(functn, opts=opts)


# --- Remaining Functions (Cleaned for JAX/CasADi only) ---

def construct_model(problem_data, cfg, supervised_learner:str, model_type:str, model_surrogate:str):
    """
    problem_data : dict    
    cfg : DictConfig
    supervised_learner : str [classification, regression]
    model_type : str
    model_surrogate : str
    """

    return surrogate_reconstruction(cfg, (supervised_learner, model_type, model_surrogate), problem_data).rebuild_model()

def generate_initial_guess(n_starts, n_d, bounds):
    n_d = len(bounds[0])
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    sobol_samples = qmc.Sobol(d=n_d, scramble=True).random(n_starts)
    return jnp.array(lower_bound) + (jnp.array(upper_bound) - jnp.array(lower_bound)) * sobol_samples



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



