import multiprocessing as mp

# TODO reconsider these imports if we are moving casadi nlp solvers to a separate module.
from casadi import SX, MX, nlpsol, Function, vertcat
import casadi
import numpy as np
from scipy.stats import qmc
from jax.experimental import jax2tf
import tensorflow.compat.v1 as tf # .compat.v1

tf.disable_v2_behavior()

def determine_batches(n_starts, num_workers):
  
    num_batches = n_starts // num_workers
    batch_size = [num_workers for _ in range(num_batches)]
    if n_starts % num_workers != 0:
      batch_size.append(n_starts % num_workers)

    return batch_size, np.cumsum(batch_size)

def create_batches(batch_size, object_iterable):
    # Create batches of data for each worker
    cumsum = np.cumsum(batch_size)
 
    return [(object_iterable[cumsum[i] : batch_size[i]]) for i in range(0, len(batch_size))]
   
    

def parallelise_batch(worker_functions, num_workers, tasks):
  input_queue = mp.Queue()
  output_queue = mp.Queue()

  # Start worker processes
  workers = [mp.Process(target=worker_function, args=(input_queue, output_queue), name=i) for i, worker_function in enumerate(worker_functions)]
  for w in workers:
      w.start()

  # Enqueue tasks
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

  return results


def worker_function(input_queue, output_queue, solver):
    # must be evalauted as a partial function
    while True:
        item, i = input_queue.get()
        if item is None:
            break
        result = solver(item)
        output_queue.put((result,i))




# TODO move the below to the solvers module
def generate_initial_guess(n_starts, n_d, bounds):
    # method for constrained nlp solution
    n_d = len(bounds[0])
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    sobol_samples = qmc.Sobol(d=n_d, scramble=True).random(n_starts)
    return np.array(lower_bound) + (np.array(upper_bound) - np.array(lower_bound)) * sobol_samples




def nlp_multi_start_casadi_eq_cons(initial_guess, objective_func, equality_constraints, bounds, cfg):
    """
    objective_func: function
    equality_constraints: function
    bounds: list
    cfg: omegaconf
    """

    lower_bound, upper_bound = bounds
    n_d = len(lower_bound)
    bnds = (lower_bound, upper_bound)

    # formatting for casadi
    constraints = lambda x: equality_constraints(x.squeeze()).reshape(-1,1)
    objective   = lambda x: objective_func(x.squeeze()).reshape(-1,1)

    # store for solutions
    solutions_store = []

    for i in range(cfg.n_starts):
        init_guess = [initial_guess[i,j] for j in range(n_d)]
        solver, solution = casadi_nlp_optimizer_eq_cons(objective, constraints, bnds, init_guess)
        if solver.stats()['success']:
            solutions_store.append(solution['f'])
            if np.array(solution['f']) <= 0: break

    try: min_obj = np.min(solutions_store)
    except: min_obj = None

    return min_obj




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
        options = {"ipopt": {"hessian_approximation": "limited-memory"}}
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


if __name__ == '__main__':
    #from multiprocessing import Pool 
    import multiprocessing as mp
    import time
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    

    @jit
    def objective(x):
        time.sleep(1)  # Simulate an expensive computation
        return jnp.sum(x**2)
  

    num_workers = 18  # Number of CPU cores
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start worker processes
    workers = [mp.Process(target=worker_function, args=(input_queue, output_queue), name=i) for i in range(num_workers)]
    for w in workers:
      w.start()

    # Enqueue tasks
    tasks = [jnp.array([1,1,2]), jnp.array([1,1,2])*2, jnp.array([1,1,2])*5]*6  # List of input data
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

    print(results)

 