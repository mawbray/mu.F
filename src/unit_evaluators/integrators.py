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
from ode import case_studies


def unit_dynamics(params, u, cfg, node   
):
    # defining the dynamics
    term = ODETerm(case_studies[cfg.case_study_dynamics][node])

    # defining the diffrax solver
    solver = dispatcher[cfg.integration.scheme[node]]

    # defining saveat 
    saveat = SaveAt(t1=True) # just return the final time step

    # define step size controller for solver
    step_size_controller = dispatcher[cfg.integration.step_size_controller]

    return diffeqsolve(
        term,
        solver,
        cfg.integration.t0,
        cfg.integration.tf,
        cfg.integration.dt0,
        u,
        args=params,
        max_steps=cfg.integration.max_steps,
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
    "pid": diffrax.PIDController(rtol=1e-7, atol=1e-7),
}

