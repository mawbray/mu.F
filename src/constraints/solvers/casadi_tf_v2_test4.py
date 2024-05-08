import unittest
import tensorflow as tf # .compat.v1
import jax.numpy as jnp
import numpy as np
from jax import jit
import casadi
from functions import casadi_nlp_solver

#class TestFunctions(unittest.TestCase):
def test_casadi_functions():
    # Define a simple TensorFlow function
    @jit
    def add(x):
        return jnp.sum(x**2).reshape(1,1)
    
    f = add


    # Define the initial guess, solver, constraints, and session
    initial_guess = np.array([1.0, 2.0]).squeeze()
    bounds = [[0.0, 0.0], [3.0, 3.0]]

    
    session = None # tf.Session()
    # Test the casadi_nlp_construction function
    solver = casadi_nlp_solver(f,f,bounds)
    solver_call = solver.casadi_nlp_construction()

    # Test the casadi_solver_call function
    solver, solution = solver_call(initial_guess)

    return solver, None

if __name__ == '__main__':
    solv, sol = test_casadi_functions()

    print(solv.stats())
    print(sol)