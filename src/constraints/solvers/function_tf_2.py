from casadi import SX, MX, nlpsol, Function, vertcat, Sparsity
import time 
import jax.numpy as jnp
import jax
import casadi 
import numpy as np
from scipy.stats import qmc
from jax.experimental import jax2tf
import tensorflow as tf 
import ray

# Enable eager execution in TensorFlow
tf.config.run_functions_eagerly(True)

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


# TODO modify casadi_nlp_optimizer_eq_cons and casadi_multi_start to handle inequality constraints as well as equality.
def casadi_nlp_optimizer_eq_cons(objective, inequality_constraints, bounds, initial_guess):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    inequality_constraint: casadi callback
    bounds: list
    initial_guess: numpy array
    Operates in a session via the casadi callbacks and tensorflow V1
    """
    n_d = len(bounds[0].squeeze())
    lb = [bounds[0].squeeze()[i] for i in range(n_d)]
    ub = [bounds[1].squeeze()[i] for i in range(n_d)]
   
    # Get the casadi callbacks required 
    cost_fn   = casadify(objective, n_d, 1)
    ineq_cons = casadify(inequality_constraints, n_d, inequality_constraints(initial_guess).squeeze().shape[0])
    
    # casadi work up
    x = MX.sym('x', n_d, 1)
    j = cost_fn(x)
    h = ineq_cons(x)

    F = Function('F', [x], [j])
    H = Function('H', [x], [h])


    # Define the box bounds
    lbx = lb
    ubx = ub

    lbg = [-np.inf] * h.size1()  # Equality constraints (0), Inequality (unbounded below)
    ubg = [0] * h.size1()  # Both equality and inequality are treated as g(x) <= 0
    

    # Define the NLP
    nlp = {'x':x , 'f':F(x), 'g':  H(x)} # 

    # Define the IPOPT solver
    options = {"ipopt": {"hessian_approximation": "limited-memory"}} # ,  , 'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter': 500
    
    solver = nlpsol('solver', 'ipopt', nlp, options)

    # Solve the NLP
    solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg) # 


    del cost_fn, ineq_cons, nlp, F,   x, j,  h, lbx, ubx, options, H, lbg, ubg,
    
      
    return solver, solution


# Introduce inequality constraint to casadi_multi_start
def casadi_multi_start(initial_guess, objective_func, inequality_constraints, bounds):
    """
    objective: casadi callback
    equality_constraints: casadi callback
    inequality_constraints: casadi callback
    bounds: list
    initial_guess: numpy array
    """
    n_starts = initial_guess.shape[0]

    # store for solutions
    solutions = []
    for i in range(n_starts):
        solver, solution = casadi_nlp_optimizer_eq_cons(objective_func,inequality_constraints, bounds, np.array(initial_guess[i,:]).squeeze())
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
    

# Convert to TensorFlow


@tf.function
def tf_objective_func(x, funct):
  
  def tf_objective_func_wrapper(x):
    return np.array(funct(jnp.array(x)))
  
  return tf.py_function(tf_objective_func_wrapper, [x], tf.float64)

# Alternative way to get the gradient using JAX and converting to TensorFlow
@tf.function
def get_gradient_jax_to_tf(x, funct):
  @jax.jit
  def jax_grad_func(x): # TODO swap this out for a jvp and return in a format in keeping with tf.gradients.
    return jax.jit(jax.grad(lambda x: funct(x).squeeze()))(x)
  
  jax_grad = jax_grad_func(jnp.array(x)).T
  return tf.convert_to_tensor(np.array(jax_grad), dtype=tf.float64)


def casadify(functn, nd, ny):
  """
  # casadify a jax function via jax-tf-casadi wrappers
  functn: a jax function 
  nd: the number of input dimensions to the function 
  session: a tf session object
  """
  opts= {}
  opts["output_dim"] = [ny, 1]
  opts["grad_dim"] = [nd, ny]

  return GPR(nd, functn, opts=opts)


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
  
  @tf.function
  def compute_gradient(self, input, adj_seed_tf):
      model = self.model
      with tf.GradientTape() as tape:
          tape.watch(input)  # Watch the input for gradients
          out_ = tf_objective_func(input, model) # Predict the output
          out_ = tf.convert_to_tensor(out_, dtype=tf.float64)  # Ensure output is a tensor
          out_ = tf.reshape(out_, (1, 1))
          grad_output = tape.gradient(out_, input, output_gradients=adj_seed_tf)
          return grad_output
      
  def eval(self, arg):
    updated_t = []
    for i, v in enumerate(self.t_in):
        # Convert CasADi input to NumPy array and reshape
        if isinstance(arg[i], (casadi.MX, casadi.DM)):
            arg_np = np.array(arg[i].full()).reshape(v.shape)
        else:
            arg_np = np.array(arg[i]).reshape(v.shape)
        updated_t.append(tf.constant(arg_np, dtype=tf.float64))  # Convert to TensorFlow tensor
    
    model = self.model
    # Case 1: Gradient calculation
    if len(arg) > 1:
    
        input = tf.reshape(updated_t[0], (1, -1))

        grad_output = get_gradient_jax_to_tf(input, model)
        #print(grad_output)
        #print("Gradient output shape:", grad_output.shape)

        selected_set = grad_output.numpy()

    # Case 2: No gradient (just the function value)
    else:
        out_, _ = self.t_out[0](tf.reshape(updated_t[0], (1, -1)))
        out_ = tf.convert_to_tensor(out_, dtype=tf.float64)  # Ensure output is a tensor

        if out_ is None:
            raise ValueError("Output from the model is None.")

        selected_set = out_.numpy().reshape((1, 1))  # Scalar output

    return [selected_set]

  # Vanilla tensorflow offers just the reverse mode AD
  def has_reverse(self,nadj): return nadj==1
  
  def get_reverse(self, nadj, name, inames, onames, opts):
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)
    adj_seed = [ tf.Variable(initializer(shape=self.sparsity_out(i).shape, dtype=tf.float64)) for i in range(self.n_out())]

    callback = TensorFlowEvaluator(self.t_in + adj_seed, [self.t_out[0]], self.model, set_init=True, opts=self.opts)
    self.refs.append(callback)

    nominal_in = self.mx_in()
    nominal_out = self.mx_out()
    adj_seed = self.mx_out()
    casadi_bal = callback.call(nominal_in + adj_seed)
    return casadi.Function(name, nominal_in+nominal_out+adj_seed, casadi_bal, inames, onames)
  


class GPR(TensorFlowEvaluator):
  def __init__(self, nd, model, opts={}):
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)
    X = tf.Variable(initializer(shape=[nd,1], dtype=tf.float64), trainable=False)

    @tf.function
    def f_k(input_dat, get_grad_val=None):
        xf_tensor = input_dat
        if(get_grad_val is not None):
            mean = tf_objective_func(xf_tensor, model)
            # grad_mean = tape.gradient(mean, xf_tensor, output_gradients=get_grad_val)
        else:
            mean = tf_objective_func(xf_tensor, model)
            grad_mean = None
        return mean, grad_mean
    
    TensorFlowEvaluator.__init__(self, [X], [f_k], model, opts=opts)
    self.counter = 0
    self.time = 0

  def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = TensorFlowEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret

if __name__ == '__main__':
    # Define a simple quadratic objective function
    def objective(x):
        return jnp.sum(x**2).reshape(1, 1)

    # Define a simple inequality constraint (x >= 0)
    def inequality_constraints(x):
        return x.reshape(1, 2)

    # Define bounds
    bounds = [jnp.array([-5, -5]), jnp.array([5, 5])]

    # Generate initial guesses
    initial_guesses = generate_initial_guess(10, 2, bounds)

    # Run the multi-start optimization
    solver, solution, n_solutions = casadi_multi_start(initial_guesses, objective, inequality_constraints, bounds)

    # Print the results
    print("Number of solutions found:", n_solutions)
    print("Optimal solution:", solution['x'])
    print("Optimal objective value:", solution['f'])