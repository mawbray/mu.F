import multiprocessing as mp
import numpy as np
import ray 

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

    def wf(input_queue, output_queue):
        while True:
            item, i = input_queue.get()
            if item is None:
                break
            result = objective(item)
            output_queue.put((result, i))

    # Start worker processes
    workers = [mp.Process(target=wf, args=(input_queue, output_queue), name=i) for i in range(num_workers)]
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

 