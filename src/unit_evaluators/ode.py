""" define ODE systems that we are interested in characterizing feasibility for """

from jax import jit 
import jax.numpy as jnp



@jit
def serial_mechanism_vc_batch_dynamics_u1(
    t: float, state: jnp.ndarray, parameters: jnp.ndarray
):
    """
    - This case study is from Pound # NOTE for Ekundayo, this is defined to give intuition as to what we need from a dynamics term
    - dynamics normalised to selection of batch time.
    - constant volume, batch mode, design variables are Temp. and Batch time
    - serial reaction mechanism 2A ->_{k1} B ->_{k2} C"""
    
    # component_concentrations
    Ca = state[0]
    Cb = state[1]
    Cc = state[2]

    # parameters
    k1, k2, tf1 = parameters[0], parameters[1], parameters[3]

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



@jit
def serial_mechanism_vc_batch_dynamics_u2(
    t: float, state: jnp.ndarray, parameters: jnp.ndarray
):
    """
    - This case study is from Pound # NOTE for Ekundayo, this is defined to give intuition as to what we need from a dynamics term
    - dynamics normalised to selection of batch time.
    - constant volume, batch mode, design variables are Temp. and Batch time
    - serial reaction mechanism 2A ->_{k1} B ->_{k2} C"""
    
    # component_concentrations
    Ca = state[0]
    Cb = state[1]
    Cc = state[2]

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

def reactor_network_4(t: float, state: jnp.ndarray, parameters: jnp.ndarray):

    # component_concentrations
    Ci = state[0]
    Cj = state[1]
    Ck = state[2]
    Cl = state[3]

    # parameters
    k1, k2, k3, k4, tf1 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    
    # normalised rates
    dCi = -(k1 + k2) * Ci
    dCj = k1 * Ci - k3 * Cj
    dCk = k2 * Ci + k3 * Cj - k4 * Ck
    dCl = - k4 * Ck 

    # differential equations
    dCi = dCi * tf1
    dCj = dCj * tf1
    dCk = dCk * tf1
    dCl = dCl * tf1

    return jnp.array([dCi, dCj, dCk, dCl])

def reactor_network_5(t: float, state: jnp.ndarray, parameters: jnp.ndarray):

    # component_concentrations
    Cb = state[0]
    Cj = state[1]
    Cm = state[2]
    Cn = state[3]

    # parameters
    k1, k2, tf1 = parameters[0], parameters[1], parameters[2]
    
    # normalised rates
    dCb = -(k1 + k2) * Cj * Cb
    dCj =  -(k1 + k2) * Cj * Cb
    dCm = k1 * Cj * Cb
    dCn = k2 * Cj * Cb

    # differential equations
    dCb = dCb * tf1
    dCj = dCj * tf1
    dCm = dCm * tf1
    dCn = dCn * tf1


    return jnp.array([dCb, dCj, dCm, dCn])





# define a dictionary that contains unit wise dynamics for each of the nodes in the graph in the case study
case_studies = {'serial_mechanism_batch': {0: serial_mechanism_vc_batch_dynamics_u1, 1: serial_mechanism_vc_batch_dynamics_u2},
                'batch_reaction_network': {0: reactor_network_4, 1: reactor_network_5}}