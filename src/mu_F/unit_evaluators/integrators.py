""" python files to define integration schemes """


# here we will implement dynamics to be used in the case studies
from typing import List

# stock imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# hydra imports
from omegaconf import DictConfig, OmegaConf

# diffrax imports
from functools import partial
import diffrax
from diffrax import ODETerm, SaveAt, diffeqsolve
from diffrax import Tsit5


# package specific imports 
from mu_F.unit_evaluators.ode import case_studies


def unit_dynamics(design_params, u, aux, decision_dependent, uncertainty_params, cfg, node):   
    """
    Here we assume that the dynamics are defined by a system of ODEs.
    This is a general function that assumes initial conditions are defined by input parameters within the extended design space 
    The design parameters, decision dependent parameters and uncertainty parameters are passed as arguments to the ODE function.
    As of yet I am not sure how to handle the case where input args provide arguments to the ODE function in a general way. 
    - NOTE to user, this is a single function which should be relatively easy to redefine to your case if the assumptions above do not hold for your system. 

    """

    if design_params.ndim < 2:
        design_params = jnp.expand_dims(design_params, axis=0)

    if decision_dependent.ndim < 2:
        decision_dependent = jnp.expand_dims(decision_dependent, axis=0)

    # defining the params to pass to the vector field
    params = jnp.hstack([design_params, decision_dependent, aux, uncertainty_params.reshape(1,-1)]).squeeze()

    # defining the dynamics
    term = ODETerm(case_studies[cfg.case_study.case_study][node])

    # defining the diffrax solver
    solver = dispatcher[cfg.model.integration.scheme]

    # defining saveat 
    saveat = SaveAt(t1=True) # just return the final time step

    # define step size controller for solver
    step_size_controller = dispatcher[cfg.model.integration.step_size_controller]
    try:
        return diffeqsolve(
        term,
        solver,
        cfg.model.integration.t0,
        cfg.model.integration.tf,
        cfg.model.integration.dt0,
        y0=u,
        args=params,
        max_steps=cfg.model.integration.max_steps,
        stepsize_controller=step_size_controller,
        saveat=saveat,
    ).ys[
        :, :
    ][-1,:]  # t x n_components
    except: # case study specific splodge
        return diffeqsolve(
        term,
        solver,
        cfg.model.integration.t0,
        cfg.model.integration.tf,
        cfg.model.integration.dt0,
        y0=jnp.hstack([u.reshape(1,-1), jnp.zeros(1).reshape(1,1)]).squeeze(),
        args=params,
        max_steps=cfg.model.integration.max_steps,
        stepsize_controller=step_size_controller,
        saveat=saveat,
    ).ys[
        :, :
    ][-1,:]  # t x n_components




# defining the dispatcher for the dynamics


dispatcher = {
    "tsit5": Tsit5(),
    "dopri8": diffrax.Dopri8(),
    "Kvaerno5": diffrax.Kvaerno5(),
    "pid": diffrax.PIDController(rtol=1e-5, atol=1e-5),
}

