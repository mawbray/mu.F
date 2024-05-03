from casadi import SX, MX, nlpsol, Function, vertcat
import casadi
import numpy as np
from scipy.stats import qmc
from jax.experimental import jax2tf
import tensorflow.compat.v1 as tf # .compat.v1

#tf.disable_v2_behavior()


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
    return np.array(lower_bound) + (np.array(upper_bound) - np.array(lower_bound)) * sobol_samples


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





def casadi_nlp_optimizer_eq_cons(objective, equality_constraints, bounds, initial_guess):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    n_d = len(bounds[0])

    with tf.Session() as session: 
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
            
        # Solve the NLP
        solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
      
    return solver, solution



def casadify(functn, nd, session):
  """
  # casadify a jax function via jax-tf-casadi wrappers
  functn: a jax function 
  nd: the number of input dimensions to the function 
  session: a tf session object
  """
  x = tf.placeholder(shape=(nd,1),dtype=tf.float64)
  fn = tf_jaxmodel_wrapper(functn)
  y = fn(x)
  fn_callback = TensorFlowEvaluator([x], [y], session)

  return fn_callback


class TensorFlowEvaluator(casadi.Callback):
  def __init__(self,t_in,t_out,session, opts={}):
    """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
      session: a tensorflow session
    """
    casadi.Callback.__init__(self)
    assert isinstance(t_in,list)
    self.t_in = t_in
    assert isinstance(t_out,list)
    self.t_out = t_out
    self.construct("TensorFlowEvaluator", opts)
    self.session = session
    self.refs = []

  def get_n_in(self): return len(self.t_in)
  def get_n_out(self): return len(self.t_out)

  def get_sparsity_in(self,i):
      return casadi.Sparsity.dense(*self.t_in[i].get_shape().as_list())

  def get_sparsity_out(self,i):
      return casadi.Sparsity.dense(*self.t_out[i].get_shape().as_list())

  def eval(self,arg):
    # Associate each tensorflow input with the numerical argument passed by CasADi
    d = dict((v,arg[i].toarray()) for i,v in enumerate(self.t_in))
    # Evaluate the tensorflow expressions
    ret = self.session.run(self.t_out,feed_dict=d)
    return ret

  # Vanilla tensorflow offers just the reverse mode AD
  def has_reverse(self,nadj): return nadj==1
  def get_reverse(self,nadj,name,inames,onames,opts):
    # Construct tensorflow placeholders for the reverse seeds
    adj_seed = [tf.placeholder(shape=self.sparsity_out(i).shape,dtype=tf.float64) for i in range(self.n_out())]
    # Construct the reverse tensorflow graph through 'gradients'
    grad = tf.gradients(self.t_out, self.t_in,grad_ys=adj_seed)
    # Create another TensorFlowEvaluator object
    callback = TensorFlowEvaluator(self.t_in+adj_seed,grad,self.session)
    # Make sure you keep a reference to it
    self.refs.append(callback)

    # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
    nominal_in = self.mx_in()
    nominal_out = self.mx_out()
    adj_seed = self.mx_out()
    return casadi.Function(name,nominal_in+nominal_out+adj_seed,callback.call(nominal_in+adj_seed),inames,onames)


def tf_jaxmodel_wrapper(jax_callable):
  return jax2tf.convert(jax_callable)

"""
utilities for JaxOpt box-constrained NLP solver
"""
import jax 
import jax.numpy as np
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
    _, solutions = jax.lax.scan(partial_jax_solver, init=None, xs=(initial_guess, np.array([1e-7] * initial_guess.shape[0])))
    now = time.time() - time_now
   
    # iterate over solutions from one of the upper level initial guesses
    assess_subproblem_solution = partial(return_most_feasible_penalty_subproblem_uncons, objective_func=objective_func)
    _, assessment = jax.lax.scan(assess_subproblem_solution, init=None, xs=solutions.params)
    
    # assessment of solutions
    arg_min = np.argmin(assessment[0], axis=0) # take the minimum objective val
    min_obj = assessment[0][arg_min]  # take the corresponding objective value
    min_grad = np.linalg.norm(assessment[1][arg_min])  # take the corresponding gradient value
    

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
    (x0, ftol) = xs

    # Define the optimization problem
    lbfgsb = LBFGSB(fun=objective_func, maxiter=1000, tol=ftol)

    problem = lbfgsb.run(x0, bounds=bounds_) # 

    return None, problem

def return_most_feasible_penalty_subproblem_uncons(init, xs, objective_func):
    # iterate over upper level initial guesses
    solution = xs
    
    # get gradients of solutions, value of objective and value of constraints
    return None, (objective_func(solution), jacfwd(objective_func)(solution))