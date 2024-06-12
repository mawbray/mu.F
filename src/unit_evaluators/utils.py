""" define any helper functions required for definition of the ODE terms"""

from jax import jit
import jax.numpy as jnp



@jit
def arrhenius_kinetics_fn(decision_params, uncertainty_params, Ea, A, R):
    temperature = decision_params[0] # temperature is always the first decision parameter
    return A * jnp.exp(-Ea / (R * temperature))


@jit
def arrhenius_kinetics_fn_2(decision_params, uncertainty_params, Ea, R):
    temperature = decision_params[0] # temperature is always the first decision parameter
    A = uncertainty_params[0]
    return A * jnp.exp(-Ea / (R * temperature))