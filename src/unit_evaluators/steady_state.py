
import jax.numpy as jnp
from unit_evaluators.implicit_fn import case_studies
from functools import partial

def unit_steady_state(design_params, u, dd_params, uncertain_params, cfg, node):   

    if design_params.ndim < 2:
        design_params = jnp.expand_dims(design_params, axis=0)

    if u.ndim < 1:
        u = jnp.expand_dims(u, axis=0)

    if u.ndim < 2:
        u = jnp.expand_dims(u, axis=0)
    
    if dd_params.ndim < 2:
        dd_params = jnp.expand_dims(dd_params, axis=0)

    # defining the params to pass to the vector field
    params = jnp.hstack([design_params, uncertain_params.reshape(1,-1)]).squeeze()

    collected_p = jnp.concatenate([u, dd_params], axis=-1)

    # defining the dynamics
    term = case_studies[cfg.case_study.case_study][node]

    return term(cfg, params, collected_p).squeeze()


