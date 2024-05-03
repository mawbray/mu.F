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
    #unittest.main()

    import jax
    import jax.numpy as jnp
    from casadi import *

    # Define your function
    def my_func(x):
        return jnp.dot(x, x)

    # Define the Jacobian-vector product using jvp
    def my_jvp(x, tangents):
        return jnp.dot(jax.grad(my_func)(x), tangents)

    # Create a pure callback for the function
    pure_callback_func = jax.pure_callback(my_func, result_shape_dtypes=[(jnp.float32,)])

    # Define symbolic variables for CasADi
    x_casadi = MX.sym('x', 2)

    # Create a callback function for CasADi that evaluates the function value and its Jacobian
    def callback_casadi(x):
        # Evaluate function value
        f_val = pure_callback_func(x)

        # Evaluate Jacobian using JVP
        jac = Function('jac', [x_casadi], [jacobian(my_func(x_casadi), x_casadi)])
        jac_val = jac(x)

        return f_val, jac_val

    # Create a CasADi optimization problem
    nlp = {'x': x_casadi, 'f': my_func(x_casadi)}
    solver_opts = {'print_level': 0}
    solver = nlpsol('solver', 'ipopt', nlp, solver_opts)

    # Initial guess
    x_guess = [1.0, 2.0]

    # Solve the optimization problem using the callback function
    sol = solver(x0=x_guess, lbx=-inf, ubx=inf, lbg=[], ubg=[], p=[], 
                lambda_=callback_casadi)

    # Extract the solution
    x_opt = sol['x']

    print("Optimal solution:", x_opt)