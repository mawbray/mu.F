
import jax.numpy as jnp
from unit_evaluators.implicit_fn import case_studies
from functools import partial

def unit_steady_state(design_params, u, uncertainty_params, cfg, node):   

    if design_params.ndim < 2:
        design_params = jnp.expand_dims(design_params, axis=0)

    # defining the params to pass to the vector field
    params = jnp.hstack([design_params, uncertainty_params.reshape(1,-1)]).squeeze()

    # defining the dynamics
    term = case_studies[cfg.case_study.case_study][node]

    return term(cfg, params, u).squeeze()


