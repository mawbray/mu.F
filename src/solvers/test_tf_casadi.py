import unittest
import jax.numpy as jnp
import numpy as np
from jax import jit
from functions import casadi_nlp_optimizer_eq_cons
import jax

class TestCasadiNlpOptimizer(unittest.TestCase):

    def test_casadi_nlp_optimizer(self):
        # Define the objective function
        @jit
        def objective(x):
            return jnp.sum(x**2).reshape(1,1)

        # Define the equality constraints
        @jit
        def equality_constraints(x):
            return jnp.sum(x).reshape(1,1) - jnp.ones(1).reshape(1,1)

        # Define the bounds
        bounds = [[-1, -1], [1, 1]]

        # Define the initial guess
        initial_guess = [0., 0.]

        # Call the casadi_nlp_optimizer method
        solver, solution = casadi_nlp_optimizer_eq_cons(objective, equality_constraints, bounds, initial_guess)

        # Assert that the result is within the bounds
        self.assertTrue((np.array(bounds[0]) <= np.array(solution['x'])).all() and (jnp.array(solution['x']) <= np.array(bounds[1])).all())

        # Assert that the solution is not None
        self.assertIsNotNone(solution)


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    
    unittest.main()
    