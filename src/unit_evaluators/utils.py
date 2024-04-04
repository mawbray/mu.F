""" define any helper functions required for definition of the ODE terms"""

from jax import jit
import jax.numpy as jnp



@jit
def arrhenius_kinetics_fn(temperature, Ea, A, R):
    return A * jnp.exp(-Ea / (R * temperature))