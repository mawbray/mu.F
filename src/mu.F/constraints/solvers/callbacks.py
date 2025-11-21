import time
from typing import Callable

import jax.numpy as jnp
import numpy as np
from casadi import DM, Callback, Function, Sparsity
from jax import jvp, jit, vjp


# --- JAX-CasADi Callback Wrappers ---
# These classes wrap JAX functions to be used within CasADi optimization problems
# They support both forward and reverse mode automatic differentiation
# Forward or reverse is selected based on the decision variable and constraints dimensions


class JaxCasADiEvaluator(Callback):
    def __init__(self, functn: Callable, nd: int, name="TensorFlowEvaluatorwReverse", opts={}):
        super().__init__()
        self.nd = nd
        self.output_shape = None
        self.refs = [] # List to hold references to reverse callbacks
        self._forward_pass = jit(functn)
        # Determine output shape and dimension by tracing (dummy run)
        dummy_x = jnp.zeros((nd, 1))
        dummy_y = self._forward_pass(dummy_x)
        self.output_shape = dummy_y.shape
        self.n_out_dim = self.output_shape[0]

        # Construct CasADi's base class
        self.construct(name, opts)

    # --- CasADi Interface Methods ---

    def get_n_in(self): return 1 # Single input variable (x)
    def get_n_out(self): return 1 # Single output variable (y)

    def get_sparsity_in(self, i):
        assert i == 0
        return Sparsity.dense(self.nd, 1)

    def get_sparsity_out(self, i):
        assert i == 0
        return Sparsity.dense(self.n_out_dim, 1)

    def eval(self, arg):
        """ Computes the forward pass y = f(x). """
        # CasADi input (arg[0]) is a DMatrix
        x_numpy = np.array(arg[0]).reshape(self.nd, 1)
        x_tf = jnp.array(x_numpy)

        # Execute the compiled tf.function
        y_tf = jnp.reshape(self._forward_pass(x_tf), (self.n_out_dim, 1))

        # Convert back to CasADi DMatrix
        return [DM([y_tf[i] for i in range(self.n_out_dim)])]

    def has_reverse(self, nadj): return nadj == 0
    
    def has_forward(self, nfwd): return nfwd == 0


class JaxCallbackForward(JaxCasADiEvaluator):
    def __init__(self, functn: Callable, nd: int, name="JaxEvaluatorwForward", opts={}):
        super().__init__(functn, nd, name, opts)
        self.counter = 0
        self.time = 0
        self.forward_pass = jit(functn)

        @jit
        def forward_propagation(primals, tangents):
            return jvp(self.forward_pass, (primals,), (tangents,))[1]
        
        self.forward_sensitivities = forward_propagation

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret
    
    def has_forward(self, nfwd): return nfwd == 1
    
    def get_forward(self, nfwd, name, inames, onames, opts):
        """ Returns a CasADi Callback that computes the forward mode AD
        of the JAX function.
        """
        assert(nfwd==1)
        nd = self.nd
        forward_sens = self.forward_sensitivities

        class ForwardEvaluator(Callback):
            def __init__(self, forward_fn, nd, n_out_dim, name, opts):
                super().__init__()
                self._forward_sensitivities = forward_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self): return 3 # x, y (nominal_out, unused), fwd_seed
            
            def get_n_out(self): return 1 # fwd_y

            def get_sparsity_in(self, i):
                if i == 0: return Sparsity.dense(self.nd, 1)        # x
                if i == 1: return Sparsity.dense(self.n_out_dim, 1) # y (nominal_out)
                if i == 2: return Sparsity.dense(self.nd, 1)        # fwd_seed
            
            def get_sparsity_out(self, i):
                return Sparsity.dense(self.n_out_dim, 1) # fwd_y

            def eval(self, arg):
                # CasADi passes: arg[0]=x, arg[1]=y (ignored), arg[2]=fwd_seed
                x_jnp = jnp.array(arg[0]).reshape(self.nd,)
                tangents = jnp.array(arg[2]).reshape(self.nd,)
                fwd_sens_list = self._forward_sensitivities(x_jnp, tangents)
                return [DM(np.array(fwd_sens_list))]

        # Instantiate the forward evaluator callback
        fwd_callback = ForwardEvaluator(forward_sens, self.nd, self.n_out_dim, f"{name}_forward", opts)
        self.refs.append(fwd_callback)

        # Construct the CasADi Function wrapper
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        fwd_seed = self.mx_in()

        # CasADi function signature: (x, y, fwd_seed) -> (fwd_y)
        return Function(
            name,
            nominal_in + nominal_out + fwd_seed,
            fwd_callback.call(nominal_in + nominal_out + fwd_seed),
            inames,
            onames
        )
    

class JaxCallbackReverse(JaxCasADiEvaluator):
    def __init__(self, functn: Callable, nd: int, name="JaxEvaluatorwReverse", opts={}):
        super().__init__(functn, nd, name, opts)
        self.counter = 0
        self.time = 0
        self.forward_pass = jit(functn)

        @jit 
        def vjp_fun(primals, tangents):
            vjp_jax = vjp(self._forward_pass, primals)[1](tangents)
            return vjp_jax[0]

        self.reverse_pass = vjp_fun

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret
    
    def has_reverse(self, nadj): return nadj == 1
    
    def get_reverse(self, nadj, name, inames, onames, opts):
        """ Returns a CasADi Callback that computes the forward mode AD
        of the JAX function.
        """
        assert(nadj==1)
        nd = self.nd
        reverse_pass = self.reverse_pass

        class ReverseEvaluator(Callback):
            def __init__(self, reverse_fn, nd, n_out_dim, name, opts):
                super().__init__()
                self._reverse_pass = reverse_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self): return 3 # x, y (nominal_out, unused), adj_seed 
            def get_n_out(self): return 1 # grad_x

            # Sparsity definitions mirror the main callback's definitions
            def get_sparsity_in(self, i):
                if i == 0: return Sparsity.dense(self.nd, 1)        # x
                if i == 1: return Sparsity.dense(self.n_out_dim, 1) # y (nominal_out)
                if i == 2: return Sparsity.dense(self.n_out_dim, 1) # adj_seed
            
            def get_sparsity_out(self, i):
                return Sparsity.dense(self.nd, 1) # grad_x  
            
            def eval(self, arg):
                # CasADi passes: arg[0]=x, arg[1]=y (ignored), arg[2]=adj_seed
                x_jnp = jnp.array(arg[0]).reshape(self.nd,1)
                adj_seed_jnp = jnp.array(arg[2]).reshape(self.n_out_dim,1)
                # Execute the jit compiled reverse
                vjp_jax = self._reverse_pass(x_jnp, adj_seed_jnp)
                # Convert back to CasADi DMatrix
                return [DM(np.array(vjp_jax))]

        # Instantiate the forward evaluator callback
        rev_callback = ReverseEvaluator(reverse_pass, nd, self.n_out_dim, f"{name}_reverse", opts)
        self.refs.append(rev_callback)

        # Construct the CasADi Function wrapper
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        rev_seed = self.mx_out()

        # CasADi function signature: (x, y, rev_seed) -> (rev_y)
        return Function(
            name,
            nominal_in + nominal_out + rev_seed,
            rev_callback.call(nominal_in + nominal_out + rev_seed),
            inames,
            onames
        )


def casadify_forward(functn, nd):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """

    return JaxCallbackForward(functn, nd)
    

def casadify_reverse(functn, nd):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """

    return JaxCallbackReverse(functn, nd)
