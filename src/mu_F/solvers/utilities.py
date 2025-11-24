import multiprocessing as mp
import numpy as np
import jax.numpy as jnp
import ray 
from functools import partial
from scipy.stats import qmc

from mu_F.surrogate.surrogate import surrogate_reconstruction
from mu_F.solvers.callbacks import casadify_forward, casadify_reverse

logger = mp.log_to_stderr()
logger.setLevel(mp.SUBDEBUG)

def determine_batches(n_starts, num_workers):
  
    num_batches = n_starts // num_workers
    batch_size = [num_workers for _ in range(num_batches)]
    if n_starts % num_workers != 0:
      batch_size.append(n_starts % num_workers)

    return batch_size, np.cumsum(batch_size)

def create_batches(batch_size, object_iterable):
    # Create batches of data for each worker
    cumsum = np.cumsum(batch_size) - batch_size[0]
 
    return [(object_iterable[cumsum[i] : cumsum[i] + batch_size[i]]) for i in range(0, len(batch_size))]
   

def parallelise_ray_batch(actors, init_guess):
    # Parallelise the batch processing
    res = [actor(init_g) for actor, init_g in zip(actors, init_guess)]
    results = ray.get(res)
    
    return [results[i] for i in range(results.shape[0])]


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
  for i in range(num_workers):
      input_queue.put((None,i))

  # Wait for all workers to finish
  for w in workers:
      w.join()

  # Collect results
  results = [None] * len(tasks)
  for i in range(num_workers):
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


def build_constraint_functions(cfg, problem_data):
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
  return g_fn


def build_objective_function(cfg, problem_data, n_d):
  # get objective function
  obj_data = problem_data['objective_func']
  n_f = len([k for k in list(obj_data.keys()) if 'f' in k])
  if n_f > 1:
    # objective requires some function composition
    obf = construct_model(obj_data['f0']['params'], cfg, supervised_learner=obj_data['f0']['model_class'], model_type=obj_data['f0']['model_type'], model_surrogate=obj_data['f0']['model_surrogate'])
    # if more than 2 functions, construct nested function
    if n_f > 2:
      obj_terms = {}
      for i in range(1,n_f-1):
        eqc = construct_model(obj_data[f'f{i}']['params'], cfg, supervised_learner=obj_data[f'f{i}']['model_class'], model_type=obj_data[f'f{i}']['model_type'], model_surrogate=obj_data[f'f{i}']['model_surrogate'])
        obj_terms[i-1] = partial(lambda x, v : eqc(x.reshape(1,-1)[:,v].reshape(-1,)).reshape(-1,1), v = jnp.array(obj_data[f'f{i}']['args']))
      # construct objective from constituent functions
      obj_in = partial(lambda x, f: jnp.hstack([f[i](x) for i in range(len(f))]), f=obj_terms)
      objective_func = partial(obj_data['obj_fn'], f1=obf, f2=obj_in)
    else:
      objective_func = partial(obj_data['obj_fn'], f1=obf)
  else:
      objective_func = lambda x: x.reshape(-1,)[obj_data['obj_fn']].reshape(1,1)

  return casadify_reverse(objective_func, n_d)


def casadify_constraints(constraints, dummy_initial_guess, n_d):
    # define the constraints function
    constraints = partial(lambda x, g: jnp.vstack([g[i](x) for i in range(len(g))]), g=constraints)
    n_g = constraints(dummy_initial_guess[0].reshape(1,-1)).reshape(-1,1).shape[0]  
    if n_g > n_d:
      constraints_fn = casadify_forward(constraints, n_d)
    else:
      constraints_fn = casadify_reverse(constraints, n_d)

    return constraints_fn, n_g


def unpack_problem_data(problem_data):
    initial_guess = problem_data['initial_guess']
    bounds = problem_data['bounds']
    lhs = problem_data['eq_lhs']
    rhs = problem_data['eq_rhs']
    n_d = initial_guess.shape[1]
    n_starts = initial_guess.shape[0]

    return initial_guess, bounds, lhs, rhs, n_d, n_starts


def unpack_results(solutions, solver, solution):
  try:
    min_obj_idx = np.argmin(np.vstack([sol_f[1]['f'] for sol_f in solutions]))
    solver_opt, solution_opt = solutions[min_obj_idx]
    n_s = len(solutions)
    return solver_opt.stats(), solution_opt, n_s
  except: 
      return solver.stats(), solution, len(solutions)
  

def clean_up(objects: list):
    for obj in objects:
        del obj