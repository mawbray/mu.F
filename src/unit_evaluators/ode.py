""" define ODE systems that we are interested in characterizing feasibility for """

from jax import jit 
import jax.numpy as jnp



@jit
def serial_mechanism_vc_batch_dynamics(
    t: float, state: jnp.ndarray, parameters: jnp.ndarray
):
    """
    - This case study is from Pound # NOTE for Ekundayo, this is defined to give intuition as to what we need from a dynamics term
    - dynamics normalised to selection of batch time.
    - constant volume, batch mode, design variables are Temp. and Batch time
    - serial reaction mechanism 2A ->_{k1} B ->_{k2} C"""
    # component_concentrations
    Ca, Cb, Cc = state[0], state[1], state[2]
    # parameters
    k1, k2, tf1 = parameters[0], parameters[1], parameters[2]

    # normalised rates
    dCa = -2 * k1 * Ca**2
    dCb = k1 * Ca**2 - k2 * Cb
    dCc = k2 * Cb

    # differential equations
    dCa = dCa * tf1
    dCb = dCb * tf1
    dCc = dCc * tf1

    # jax.debug.print('dC {x}', x=[dCa, dCb, dCc])

    return jnp.array([dCa, dCb, dCc])



# define a dictionary that contains unit wise dynamics for each of the nodes in the graph in the case study
case_studies = {'serial_batch': {0: serial_mechanism_vc_batch_dynamics, 1: serial_mechanism_vc_batch_dynamics}}