from time import time
from casadi import Callback, DM, Sparsity
import tensorflow as tf
import casadi as ca
from jax import jit, jacrev, numpy as jnp
from jax.experimental import jax2tf
import numpy as np


def tf_jaxmodel_wrapper(jax_callable):
    """
    Converts a JAX callable function into a TensorFlow callable.
    This is necessary to define the TensorFlow graph that CasADi will use.
    """
    return jax2tf.convert(jax_callable, enable_xla=False)

# --- TensorFlow V2 CasADi Evaluator Class ---

class TensorFlowEvaluator(ca.Callback):
    """
    A CasADi Callback that uses TensorFlow V2's tf.function and tf.GradientTape
    to compute forward evaluation and reverse-mode AD (Jacobian-vector product).
    """
    def __init__(self, functn, nd, name="TensorFlowEvaluator", opts={}):
        ca.Callback.__init__(self)
        self.nd = nd
        self.tf_fn = tf_jaxmodel_wrapper(functn)
        self.output_shape = None
        self.refs = [] # List to hold references to reverse callbacks

        # 1. Compile the forward pass into a concrete tf.function
        @tf.function(input_signature=[tf.TensorSpec(shape=(nd, 1), dtype=tf.float64)])
        def forward_pass(x):
            return self.tf_fn(x)

        self._forward_pass_tf = forward_pass

        # 2. Determine output shape and dimension by tracing (dummy run)
        dummy_x = tf.constant(np.zeros((nd, 1)), dtype=tf.float64)
        dummy_y = self._forward_pass_tf(dummy_x)
        self.output_shape = dummy_y.shape.as_list()
        self.n_out_dim = self.output_shape[0]

        # 3. Call CasADi's base class constructor
        self.construct(name, opts)


    # --- CasADi Interface Methods ---

    def get_n_in(self): return 1 # Single input variable (x)
    def get_n_out(self): return 1 # Single output variable (y)

    def get_sparsity_in(self, i):
        assert i == 0
        return ca.Sparsity.dense(self.nd, 1)

    def get_sparsity_out(self, i):
        assert i == 0
        return ca.Sparsity.dense(self.n_out_dim, 1)

    def eval(self, arg):
        """ Computes the forward pass y = f(x). """
        # CasADi input (arg[0]) is a DMatrix
        x_numpy = np.array(arg[0]).reshape(self.nd, 1)
        x_tf = tf.constant(x_numpy, dtype=tf.float64)

        # Execute the compiled tf.function
        y_tf = self._forward_pass_tf(x_tf)

        # Convert back to CasADi DMatrix
        return [ca.DM(y_tf.numpy())]

    # --- Reverse AD Implementation ---

    def has_reverse(self, nadj): return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        nd = self.nd
        n_out_dim = self.n_out_dim

        # 1. Define the compiled Reverse pass function (Jacobian-vector product)
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(nd, 1), dtype=tf.float64),      # x (nominal_in)
            tf.TensorSpec(shape=(n_out_dim, 1), dtype=tf.float64) # adj_seed
        ])
        def reverse_pass(x, adj_seed):
            """Computes grad = grad_x(f(x)) * adj_seed using GradientTape."""
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                y = self.tf_fn(x) # Forward pass
            # Compute the vector-Jacobian product (VJP)
            grad = tape.gradient(y, x, output_gradients=adj_seed)
            # Ensure the output is a list, as CasADi expects a list of outputs
            return [grad]

        # 2. Define a specialized callback class to wrap the reverse function
        # This wrapper adheres to CasADi's required signature: (x, y, adj_seed) -> (grad_x)
        class ReverseEvaluator(ca.Callback):
            def __init__(self, reverse_fn, nd, n_out_dim, name, opts):
                ca.Callback.__init__(self)
                self._reverse_pass_tf = reverse_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self): return 3 # x, y (nominal_out, unused), adj_seed
            def get_n_out(self): return 1 # grad_x

            # Sparsity definitions mirror the main callback's definitions
            def get_sparsity_in(self, i):
                if i == 0: return ca.Sparsity.dense(self.nd, 1)        # x
                if i == 1: return ca.Sparsity.dense(self.n_out_dim, 1) # y (nominal_out)
                if i == 2: return ca.Sparsity.dense(self.n_out_dim, 1) # adj_seed
            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(self.nd, 1) # grad_x

            def eval(self, arg):
                # CasADi passes: arg[0]=x, arg[1]=y (ignored), arg[2]=adj_seed
                x_tf = tf.constant(np.array(arg[0]).reshape(self.nd, 1), dtype=tf.float64)
                adj_seed_tf = tf.constant(np.array(arg[2]).reshape(self.n_out_dim, 1), dtype=tf.float64)

                # Execute the compiled reverse tf.function
                grad_tf_list = self._reverse_pass_tf(x_tf, adj_seed_tf)

                # Convert back to CasADi DMatrix
                return [ca.DM(grad_tf_list[0].numpy())]

        # 3. Instantiate the reverse callback and keep a reference
        rev_callback = ReverseEvaluator(reverse_pass, self.nd, self.n_out_dim, f"{name}_reverse", opts)
        self.refs.append(rev_callback)

        # 4. Construct the CasADi Function wrapper
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()

        # CasADi function signature: (x, y, adj_seed) -> (grad_x)
        return ca.Function(
            name,
            nominal_in + nominal_out + adj_seed,
            rev_callback.call(nominal_in + nominal_out + adj_seed),
            inames,
            onames
        )
    
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
        self.jitted_model = jit(self.model)
        self.jitted_grad_func = jit(
            jacrev(self.jitted_model)
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


def scalarfn_casadify(functn, nd, ny=1):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """
    opts = {"output_dim": [1, ny], "grad_dim": [nd, ny]}
    return ScalarFn(functn, opts=opts)
    

def vectorfn_casadify(functn, nd):
    """
    # casadify a jax function via jax-tf-casadi wrappers using TensorFlow v2 tf.function.
    functn: a jax function
    nd: the number of input dimensions to the function
    """
    fn_callback = TensorFlowEvaluator(functn, nd)
    return fn_callback


if __name__ == "__main__":

    # --- Example Usage ---

    # 1. Define a simple JAX function
    def jax_func(x):
        # x is (nd, 1)
        return jnp.array([[jnp.sin(x[0, 0]) * x[1, 0]],
                        [jnp.cos(x[1, 0]**2) + x[0, 0]]])

    # 2. Input dimension
    INPUT_DIM = 2

    # 3. Create the CasADi callback object
    tf_callback = casadify(jax_func, INPUT_DIM)

    # 4. Create an MX symbol for CasADi optimization/manipulation
    x_sym = ca.MX.sym('x', INPUT_DIM, 1)

    # 5. Create the CasADi Function that uses the callback
    F = ca.Function('F', [x_sym], [tf_callback(x_sym)])

    # 6. Test forward evaluation
    x_test = ca.DM([1.0, 2.0])
    y_test = F(x_test)
    print(f"Input x:\n{x_test}")
    print(f"\nForward Evaluation F(x):\n{y_test}")
    # Expected: [[sin(1)*2], [cos(4)+1]] approx: [[1.6829], [0.3463]]

    # 7. Test Reverse AD (Jacobian calculation)
    # We can use CasADi to calculate the Jacobian based on the callback's reverse function
    J = ca.jacobian(F(x_sym), x_sym)
    Jac_func = ca.Function('Jac', [x_sym], [J])

    # Evaluate the Jacobian
    J_test = Jac_func(x_test)
    print(f"\nJacobian J(x):\n{J_test}")

    # Expected Jacobian at x=[1, 2]:
    # J11 = d/dx0 (sin(x0)x1) = cos(x0)x1 = cos(1)*2 approx 1.0806
    # J12 = d/dx1 (sin(x0)x1) = sin(x0) = sin(1) approx 0.8415
    # J21 = d/dx0 (cos(x1^2)+x0) = 1
    # J22 = d/dx1 (cos(x1^2)+x0) = -sin(x1^2) * 2*x1 = -sin(4) * 4 approx 3.0279
    # Result: [[1.0806, 0.8415], [1.0000, 3.0279]]