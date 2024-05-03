import tensorflow as tf
import casadi

class TensorFlowEvaluator(casadi.Callback):
    def __init__(self, t_in, t_out, opts={}):
        """
        t_in: list of inputs (tensorflow placeholders)
        t_out: list of outputs (tensors dependeant on those placeholders)
        """
        casadi.Callback.__init__(self)
        assert isinstance(t_in, list)
        self.t_in = t_in
        assert isinstance(t_out, list)
        self.t_out = t_out
        self.construct("TensorFlowEvaluator", opts)
        self.refs = []

    def get_n_in(self): return len(self.t_in)
    def get_n_out(self): return len(self.t_out)

    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.t_in[i].shape.as_list())

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.t_out[i].shape.as_list())

    def eval(self, arg):
        # Associate each tensorflow input with the numerical argument passed by CasADi
        d = dict((v, arg[i].toarray()) for i, v in enumerate(self.t_in))
        # Evaluate the tensorflow expressions
        ret = [t.numpy() for t in self.t_out]
        return ret

    # Vanilla tensorflow offers just the reverse mode AD
    def has_reverse(self, nadj): return nadj == 1
    def get_reverse(self, nadj, name, inames, onames, opts):
        # Construct tensorflow placeholders for the reverse seeds
        adj_seed = [tf.Variable(tf.zeros(self.sparsity_out(i).shape), dtype=tf.float64) for i in range(self.n_out())]
        # Construct the reverse tensorflow graph through 'gradients'
        grad = tf.gradients(self.t_out, self.t_in, grad_ys=adj_seed)
        # Create another TensorFlowEvaluator object
        callback = TensorFlowEvaluator(self.t_in + adj_seed, grad)
        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return casadi.Function(name, nominal_in + nominal_out + adj_seed, callback.call(nominal_in + adj_seed), inames, onames)
    
import unittest
import tensorflow as tf
import numpy as np
import casadi

class TestTensorFlowEvaluator(unittest.TestCase):
    def test_eval(self):
        # Define a simple TensorFlow function
        @tf.function
        def add(x, y):
            return x + y

        # Define the inputs and outputs
        x = tf.Variable(0.0)
        y = tf.Variable(0.0)
        t_in = [x, y]
        t_out = [add(x, y)]

        # Create the TensorFlowEvaluator
        evaluator = TensorFlowEvaluator(t_in, t_out)

        # Create a CasADi function from the TensorFlowEvaluator
        f = casadi.Function('f', [x, y], [evaluator])

        # Test the eval method
        result = f(1.0, 2.0)
        self.assertEqual(result, 3.0)

if __name__ == '__main__':
    unittest.main()