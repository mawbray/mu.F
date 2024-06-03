import unittest
import tensorflow as tf # .compat.v1
import jax.numpy as jnp
import numpy as np
from jax import jit
import casadi
from functions import casadi_nlp_optimizer_eq_cons


def test_casadi_ms_solver_call():
    # Define your inputs here
    def objective_func(x):
        return jnp.sum(jnp.array(x)**2).reshape(-1,1)

    def equality_constraints(x):
        return jnp.array(x - 1).reshape(-1,1)

    bounds = [np.array([-1.0, -1.0]).squeeze(), np.array([1.0, 1.0]).squeeze()]
    initial_guess = np.array([0.0, 0.0]).reshape(-1)

    # Call the function with your inputs
    nlpsolver, result = casadi_nlp_optimizer_eq_cons(objective_func, equality_constraints, bounds, initial_guess)

    return nlpsolver, result

if __name__ == '__main__':
    solver, result = test_casadi_ms_solver_call()
