from casadi import SX, MX, nlpsol, Function, vertcat, Sparsity
import casadi 
import numpy as np
from scipy.stats import qmc
from jax.experimental import jax2tf
import tensorflow as tf 
import multiprocessing as mp

from constraints.solvers.utilities import determine_batches, create_batches, parallelise_batch, worker_function



"""
utilities for Casadi NLP solver with equality constraints
"""

def generate_initial_guess(n_starts, n_d, bounds):
    """
    Here we have defined a Sobol sequence to generate the initial guesses for the NLP solver
    n_starts: int
    n_d: int
    bounds: list
    - NOTE: this could be tidier if we loaded the sampler onto the solvers class from the samplers module.
    """
    # method for constrained nlp solution
    n_d = len(bounds[0])
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    sobol_samples = qmc.Sobol(d=n_d, scramble=True).random(n_starts)
    return jnp.array(lower_bound) + (jnp.array(upper_bound) - jnp.array(lower_bound)) * sobol_samples


def nlp_multi_start_casadi_eq_cons(initial_guess, objective_func, equality_constraints, bounds, solver):
    """
    objective_func: function
    equality_constraints: function
    bounds: list

    """
    lower_bound, upper_bound = bounds
    n_d = len(lower_bound)
    bnds = (lower_bound, upper_bound)
    n_starts = initial_guess.shape[0]

    # formatting for casadi
    constraints = lambda x: equality_constraints(x.squeeze()).reshape(-1,1)
    objective   = lambda x: objective_func(x.squeeze()).reshape(-1,1)

    # store for solutions
    solutions_store = []

    for i in range(n_starts):
        init_guess = [initial_guess[i,j] for j in range(n_d)]
        solver, solution = casadi_nlp_optimizer_eq_cons(objective, constraints, bnds, init_guess, solver)
        if solver.stats()['success']:
            solutions_store.append((solver,solution))
            if np.array(solution['f']) <= 0: break

    try: 
        min_obj_idx = np.argmin(np.vstack([sol_f[1]['f'].reshape(1,-1) for sol_f in solutions_store]))
        solver_opt, solution_opt = solutions_store[min_obj_idx]   
        return solver_opt, solution_opt
    
    except: 
       return None, None


def get_session():
    return tf.Session()


def casadi_nlp_optimizer_eq_cons(objective, equality_constraints, bounds, initial_guess):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    Operates in a session via the casadi callbacks and tensorflow V1
    """
    n_d = len(bounds[0])
    session = tf.Session()

    with session: 
        # Get the casadi callbacks required 
        cost_fn   = casadify(objective, n_d, session)
        eq_cons   = casadify(equality_constraints, n_d, session)
        
        # casadi work up
        x = MX.sym('x', n_d,1)
        j = cost_fn(x)
        g = eq_cons(x)

        F = Function('F', [x], [j])
        G = Function('G', [x], [g])


        # Define the box bounds
        lbx = bounds[0] 
        ubx = bounds[1]

        # Define the bounds for the equality constraints
        lbg = 0
        ubg = 0

        # Define the NLP
        nlp = {'x':x , 'f':F(x), 'g': G(x)}

        # Define the IPOPT solver
        options = {"ipopt": {"hessian_approximation": "limited-memory"}} #'ipopt.print_level':0, 'print_time':0}
      
        solver = nlpsol('solver', 'ipopt', nlp, options)

    with session:
        # Solve the NLP
        solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
      
    return solver, solution


def casadi_nlp_construction(objective_func, equality_constraints, bounds):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    n_d = len(bounds[0])
    print(n_d)

    # Get the casadi callbacks required 
    cost_fn   = casadifyV2(objective_func, n_d)
    eq_cons   = casadifyV2(equality_constraints, n_d)
    
    # casadi work up
    x = MX.sym('x', n_d, 1)
    j = cost_fn(x)
    g = eq_cons(x)

    F = Function('F', [x], [j])
    G = Function('G', [x], [g])

    # Define the box bounds
    lbx = bounds[0] 
    ubx = bounds[1]

    # Define the bounds for the equality constraints
    lbg = 0
    ubg = 0

    # Define the NLP
    nlp = {'x':x , 'f':F(x), 'g': G(x)}

    # Define the IPOPT solver
    options = {"ipopt": {"hessian_approximation": "limited-memory"}} #'ipopt.print_level':0, 'print_time':0}
  
    solver = nlpsol('solver', 'ipopt', nlp, options)


    
    return solver, [lbx, ubx, lbg, ubg], nlp

def casadi_solver_call(solver, constraints, initial_guess, nlp):
    """
    solver: casadi solver
    bounds: generator
    initial_guess: numpy array
    """
    [lbx, ubx, lbg, ubg] = constraints

    solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return solver, solution


def casadi_multi_start_solver_call(objective_func, equality_constraints, bounds, initial_guess, device_count):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    n_d = len(bounds[0])
    n_starts = initial_guess.shape[0]
    print(n_d)

    # Get the casadi callbacks required 
    cost_fn   = casadifyV2(objective_func, n_d)
    eq_cons   = casadifyV2(equality_constraints, n_d)
    
    # casadi work up
    x = MX.sym('x', n_d, 1)
    j = cost_fn(x)
    g = eq_cons(x)

    F = Function('F', [x], [j])
    G = Function('G', [x], [g])

    # Define the box bounds
    lbx = bounds[0] 
    ubx = bounds[1]

    # Define the bounds for the equality constraints
    lbg = 0
    ubg = 0

    # Define the NLP
    nlp = {'x':x , 'f':F(x), 'g': G(x)}

    # Define the IPOPT solver
    options = {"ipopt": {"hessian_approximation": "limited-memory"}} #'ipopt.print_level':0, 'print_time':0}
    solver = nlpsol('solver', 'ipopt', nlp, options)


    def optimize(x0):
        res = solver(x0=[i for i in x0.squeeze()], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        if res.stats()["success"]:
            return float(res["f"]), np.array(res["x"])
        else:
            return np.inf, np.zeros(n_d)
    
    # definition of the worker function
    wf = partial(worker_function, solver=optimize)

    # Start worker processes
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    num_workers = min(device_count, n_starts)
    workers = [mp.Process(target=wf, args=(input_queue, output_queue), name=i) for i in range(num_workers)]
    for w in workers:
      w.start()

    # Enqueue tasks
    tasks = [initial_guess[i,:].squeeze() for i in range(n_starts)]  # List of input data
    for i, task in enumerate(tasks):
      input_queue.put((task,i))

    # Signal workers to terminate
    for _ in range(num_workers):
      input_queue.put((None,i))

    # Wait for all workers to finish
    for w in workers:
      w.join()

    # Collect results
    results = [None] * len(tasks)
    while not output_queue.empty():
      result, i = output_queue.get()
      results[i] = result

    minima, _ = min(results, key=lambda x: x[0])

    if minima == np.inf:
       return None, None
    else:
       _, optimal_x = results[np.argmin([res[0] for res in results])]
       return minima, optimal_x


def evaluate_casadi_nlp_ms(initial_guess, objective_func, equality_constraints, bounds, device_count):
    """
    objective_func: function
    equality_constraints: function
    bounds: list

    """

    # store for solutions
    result_f, result_x = casadi_multi_start_solver_call(objective_func, equality_constraints, bounds, initial_guess, device_count)

    return result_f, result_x
   

def casadifyV2(functn, nd):
  """
  # casadify a jax function via jax-tf-casadi wrappers
  functn: a jax function 
  nd: the number of input dimensions to the function 
  """
  opts = {}
  opts["output_dim"] = [1, 1]
  opts["grad_dim"] = [1, nd]
  fn = tf.function(tf_jaxmodel_wrapper(functn), jit_compile=True, autograph=False)

  fn_callback = casadi_model(nd, fn, opts=opts)

  return fn_callback



def tf_jaxmodel_wrapper(jax_callable):
  return jax2tf.convert(jax_callable)

"""
utilities for JaxOpt box-constrained NLP solver
"""
import jax 
import jax.numpy as jnp
from jax import grad, jit, vmap, config, jacfwd, jacrev
from jaxopt import LBFGSB
from functools import partial 
import time


def multi_start_solve_bounds_nonlinear_program(initial_guess, objective_func, bounds_):
    """
    objective is a partial function which just takes as input the decision variables and returns the objective value
    constraints is a vector valued partial function which just take as input the decision variables and return the constraint value, in the form np.inf \leq g(x) \leq 0
    bounds is a list with 
    """
    solutions = []


    partial_jax_solver = jit(partial(solve_nonlinear_program_bounds_jax_uncons, objective_func=objective_func, bounds_=bounds_))

    # iterate over upper level initial guesses
    time_now  = time.time()
    _, solutions = jax.lax.scan(partial_jax_solver, init=None, xs=(initial_guess))
    now = time.time() - time_now
   
    # iterate over solutions from one of the upper level initial guesses
    assess_subproblem_solution = partial(return_most_feasible_penalty_subproblem_uncons, objective_func=objective_func)
    _, assessment = jax.lax.scan(assess_subproblem_solution, init=None, xs=solutions.params)
    
    # assessment of solutions
    arg_min = jnp.argmin(assessment[0], axis=0) # take the minimum objective val
    min_obj = assessment[0][arg_min]  # take the corresponding objective value
    min_grad = jnp.linalg.norm(assessment[1][arg_min])  # take the corresponding l2 norm of objective gradient
    

    return min_obj.squeeze(), min_grad, solutions[1].error[arg_min].squeeze()


def solve_nonlinear_program_bounds_jax_uncons(init, xs, objective_func, bounds_):
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
    lbfgsb = LBFGSB(fun=objective_func, maxiter=1000, use_gamma=True, verbose=False, linesearch="backtracking", decrease_factor=0.8)

    problem = lbfgsb.run(x0, bounds=bounds_) # 

    return None, problem

def return_most_feasible_penalty_subproblem_uncons(init, xs, objective_func):
    # iterate over upper level initial guesses
    solution = xs
    
    # get gradients of solutions, value of objective and value of constraints
    return None, (objective_func(solution), jacfwd(objective_func)(solution))



class TensorFlowEvaluator(casadi.Callback):
  def __init__(self, t_in, t_out, model, set_init=False, opts={}):
  
    self.set_init = set_init
    self.opts = opts
    casadi.Callback.__init__(self)
    assert isinstance(t_in,list)
    self.t_in = t_in
    assert isinstance(t_out, list)
    self.t_out = t_out
    self.output_shapes = []
    self.construct("TensorFlowEvaluator", {})
    self.refs = []
    self.model = model
    

  def get_n_in(self): return len(self.t_in)

  def get_n_out(self): return len(self.t_out)

  def get_sparsity_in(self, i):
      tesnor_shape = self.t_in[i].shape
      return Sparsity.dense(tesnor_shape[0], tesnor_shape[1])

  def get_sparsity_out(self, i):
      if(i == 0 and self.set_init is False):
        tensor_shape = [self.opts["output_dim"][0], self.opts["output_dim"][1]]
      elif (i == 0 and self.set_init is True):
        tensor_shape = [self.opts["grad_dim"][0], self.opts["grad_dim"][1]]
      else:
         tensor_shape = [self.opts["output_dim"][0], self.opts["output_dim"][1]]
      return Sparsity.dense(tensor_shape[0], tensor_shape[1])

  def eval(self, arg):
    updated_t = []
    for i,v in enumerate(self.t_in):
        updated_t.append(tf.Variable(arg[i].toarray()))
    if(len(updated_t) == 1):
      out_, grad_estimate = self.t_out[0](tf.convert_to_tensor(updated_t[0].numpy(), dtype=tf.float32))
    else:
      out_, grad_estimate = self.t_out[0](tf.convert_to_tensor(updated_t[0].numpy(), dtype=tf.float32), tf.convert_to_tensor(updated_t[1].numpy(), dtype=tf.float32))

    if(len(updated_t) == 1):
          selected_set =  out_.numpy() 
    else:
          selected_set = grad_estimate.numpy()
    return [selected_set]

  # Vanilla tensorflow offers just the reverse mode AD
  def has_reverse(self,nadj): return nadj==1
  
  def get_reverse(self, nadj, name, inames, onames, opts):
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)
    adj_seed = [tf.Variable(initializer(shape=self.sparsity_out(i).shape, dtype=tf.float32)) for i in range(self.n_out())]

    callback = TensorFlowEvaluator(self.t_in + adj_seed, [self.t_out[0]], self.model, set_init=True, opts=self.opts)
    self.refs.append(callback)

    nominal_in = self.mx_in()
    nominal_out = self.mx_out()
    adj_seed = self.mx_out()
    casadi_bal = callback.call(nominal_in + adj_seed)
    return Function(name, nominal_in+nominal_out+adj_seed, casadi_bal, inames, onames)

class casadi_model(TensorFlowEvaluator):
  def __init__(self, nd, model, opts={}):
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)
    X = tf.Variable(initializer(shape=[1,nd], dtype=tf.float32), trainable=False)

    @tf.function
    def f_k(input_dat, get_grad_val=None):
        xf_tensor = input_dat
        if(get_grad_val is not None):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(xf_tensor)
                pred = model(xf_tensor)
            grad_pred = tape.gradient(pred, xf_tensor, output_gradients=get_grad_val)
        else:
            pred = model(xf_tensor)
            grad_pred = None
        return pred, grad_pred
    
    TensorFlowEvaluator.__init__(self, [X], [f_k], model, opts=opts)

  def eval(self,arg):
    ret = TensorFlowEvaluator.eval(self, arg)
    return [ret]