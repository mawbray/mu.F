from casadi import SX, MX, nlpsol, Function, vertcat, Sparsity
import casadi 
import numpy as np
from scipy.stats import qmc
import jax.numpy as jnp
from jax.experimental import jax2tf
import multiprocessing as mp
import tensorflow.compat.v1 as tf # .compat.v1
import ray
from functools import partial

from constraints.solvers.surrogate.surrogate import surrogate_reconstruction

tf.disable_v2_behavior()



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
        del solutions_store
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
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]
    #tf.keras.backend.clear_session()
    tf.reset_default_graph()
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
        lbx = lb
        ubx = ub

        # Define the bounds for the equality constraints
        lbg = 0
        ubg = 0

        # Define the NLP
        nlp = {'x':x , 'f':F(x), 'g': G(x)}

        # Define the IPOPT solver
        options = {"ipopt": {"hessian_approximation": "limited-memory"}, 'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 150} # , 
      
        solver = nlpsol('solver', 'ipopt', nlp, options)

        # Solve the NLP
        solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
  
    session.close()
    
    del session, cost_fn, eq_cons, nlp, F, G, x, j, g, lbx, ubx, lbg, ubg, options
    
      
    return solver, solution




def casadi_multi_start(initial_guess, objective_func, constraints, bounds):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    n_starts = initial_guess.shape[0]

    # store for solutions
    solutions = []
    for i in range(n_starts):
        solver, solution = casadi_nlp_optimizer_eq_cons(objective_func, constraints, bounds, np.array(initial_guess[i,:]).squeeze())
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
        return solver, solution, len(solutions)

def construct_model(problem_data, cfg, supervised_learner:str, model_type:str, model_surrogate:str):
    """
    problem_data : dict    
    cfg : DictConfig
    supervised_learner : str [classification, regression]
    model_type : str
    model_surrogate : str
    """

    return surrogate_reconstruction(cfg, (supervised_learner, model_type, model_surrogate), problem_data).rebuild_model()
   

@ray.remote(num_cpus=1)
def ray_casadi_multi_start(problem_id, problem_data, cfg):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    # TODO update this to handle the case where the problem_data is a dictionary and the contraints are inequality constraints
    initial_guess, bounds = \
      problem_data['initial_guess'], problem_data['bounds']
    n_starts = initial_guess.shape[0]

    # get constraint functions and define masking of the inputs
    g_fn = {}
    for i, cons_data in enumerate(problem_data['constraints'].values()):
      fn = construct_model(cons_data['params'], cfg, 
                            supervised_learner=cons_data['model_class'],
                            model_class=cons_data['model_subclass'],
                            model_surrogate=cons_data['model_surrogate'])
      if problem_data['uncertain_params'] == None:
        g_fn[i] = partial(lambda x, v : fn(x.reshape(1,-1)[:,v]).reshape(-1,1), v = cons_data['args'])
      else:
         raise NotImplementedError("Uncertain parameters not yet implemented for inequality constraints")
    
    # define the constraints function
    constraints = partial(lambda x, g: jnp.vstack([g[i](x) for i in range(len(g))]), g=g_fn)

    # get objective function
    obj_data = problem_data['objective_func']
    n_f = len([k for k in list(obj_data.keys()) if 'f' in k])
    obf = construct_model(obj_data['f0']['params'], cfg, supervised_learner=obj_data['f0']['model_class'], model_class=obj_data['f0']['model_subclass'], model_surrogate=obj_data['f0']['model_surrogate'])
    if n_f > 1:
      obj_terms = {}
      for i in range(1,n_f):
        eqc = construct_model(obj_data[f'f{i}']['params'], cfg, supervised_learner=obj_data[f'f{i}']['model_class'], model_class=obj_data[f'f{i}']['model_subclass'], model_surrogate=obj_data[f'f{i}']['model_surrogate'])
        obj_terms[i] = partial(lambda x, v : eqc(x.reshape(1,-1)[:,v]).reshape(-1,1), v = obj_data[f'f{i}']['args'])
      # construct objective from constituent functions
      obj_in = partial(lambda x, g: jnp.hstack([g[i](x) for i in range(len(g))]), g=obj_terms)
      objective_func = partial(obj_data['obj_fn'], f1=obf, f2=obj_in)
    else:
      objective_func = partial(obj_data['obj_fn'], f1=obf)
    # TODO update this to import lhs and rhs from the problem_data
    # store for solutions
    solutions = []
    for i in range(n_starts):
        solver, solution = casadi_nlp_optimizer_eq_cons(objective_func, constraints, bounds, np.array(initial_guess[i,:]).squeeze(), lhs, rhs)
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
import jax.numpy as jnp
from jax import grad, jit, vmap, config, jacfwd, jacrev
from jaxopt import LBFGSB
from functools import partial 
import time


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
    _, solutions = jax.lax.scan(partial_jax_solver, init=None, xs=(initial_guess))
    now = time.time() - time_now

    
   
    # iterate over solutions from one of the upper level initial guesses
    assess_subproblem_solution = partial(return_most_feasible_penalty_subproblem_uncons, objective_func=objective_func)
    _, assessment = jax.lax.scan(assess_subproblem_solution, init=None, xs=solutions.params)
    

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




# code graveyard 


'''
def casadi_ms_solver_call(objective_func, equality_constraints, bounds, initial_guess, n_g):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    # Define the box bounds
    lbx = [bounds[0][i] for i in range(bounds[0].shape[0])] 
    ubx = [bounds[1][i] for i in range(bounds[1].shape[0])] 

    n_d = len(lbx)
    n_starts = initial_guess.shape[0]

    # Get the casadi callbacks required 
    cost_fn   = casadifyV2(objective_func, n_d, 1)
    eq_cons   = casadifyV2(equality_constraints, n_d, n_g)
    
    # casadi work up
    x = MX.sym('x', n_d, 1)
    j = cost_fn(x)
    g = eq_cons(x)

    F = Function('F', [x], [j])
    G = Function('G', [x], [g])

    # Define the bounds for the equality constraints
    lbg = 0
    ubg = 0

    # Define the NLP
    nlp = {'x':x , 'f':F(x), 'g': G(x)}

    # Define the IPOPT solver
    options = {"ipopt": {"hessian_approximation": "limited-memory"}} #'ipopt.print_level':0, 'print_time':0}
    nlpsolver = nlpsol('solver', 'ipopt', nlp, options)

    result = nlpsolver(x0 =[initial_guess.squeeze()[i] for i in range(n_d)], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return nlpsolver, result


def casadi_multi_start_solver_call(objective_func, equality_constraints, bounds, initial_guess, device_count,):
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
    nlpsolver = nlpsol('solver', 'ipopt', nlp, options)

    class Solver(object):
      def __init__(self, x0):
          self.x0 = x0
          self.nlpsolver = nlpsolver

      def solve(self):
          self.sol = self.nlpsolver(x0 = self.x0)

    def solve(obj, q):
      obj.solve()
      q.put(obj)
      

    # Start worker processes    
    num_workers = min(1,min(device_count, n_starts))
    # definition of the worker function
    tasks = [initial_guess[i,:].squeeze() for i in range(num_workers)]  # List of input data
    q = mp.Queue()
    solver_i = [Solver(task) for task in tasks]
    workers = [mp.Process(target = solve, args = (solver_i[i], q)) for i in range(num_workers)]
    
    for w in workers:
      w.start()

    # Enqueue tasks
    
    q.empty() # returns False

    result = q.get() # Fails with the error given below

    """# Wait for all workers to finish
    for w in workers:
      w.join()

    # Collect results
    results = [None] * len(tasks)
    while not q.empty():
      result, i = q.get()
      results[i] = result
    """
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
   

def casadifyV2(functn, nd, ng):
  """
  # casadify a jax function via jax-tf-casadi wrappers
  functn: a jax function 
  nd: the number of input dimensions to the function 
  """
  opts = {}
  opts["output_dim"] = [1, ng]
  opts["grad_dim"] = [ng, nd]
  fn = tf.function(tf_jaxmodel_wrapper(functn), jit_compile=True, autograph=False)

  fn_callback = casadi_model(nd, fn, opts=opts)

  return fn_callback


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
  


def casadi_nlp_construction(objective_func, equality_constraints, bounds):
    
    objective: casadi callback
    equality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    
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
    
    solver: casadi solver
    bounds: generator
    initial_guess: numpy array
    
    [lbx, ubx, lbg, ubg] = constraints

    solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return solver, solution

'''
