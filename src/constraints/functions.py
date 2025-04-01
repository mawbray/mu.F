""" case study specific functions """

from jax import jit, vmap
import jax.numpy as jnp
from functools import partial

# ----------------------------------------------------------------------------- #
# ---------------------------- Serial Mechanism Batch ------------------------- #
# ----------------------------------------------------------------------------- #

# --- critical quality attribute constraints --- #
@partial(jit, static_argnums=(1))
def purity_b(dynamic_profile, cfg):
    pb = dynamic_profile[1] / jnp.sum(dynamic_profile[:])
    return pb


@partial(jit, static_argnums=(1))
def purity_c(dynamic_profile, cfg):
    return dynamic_profile[2] / jnp.sum(dynamic_profile[:])


# --- constraint indicators --- #

@partial(jit, static_argnums=(1))
def purity_unit_2_lb(dynamic_profile, cfg):
    return purity_b(dynamic_profile, cfg) - cfg.constraint.purity_u2


@partial(jit, static_argnums=(1))
def purity_unit_1_ub(dynamic_profile, cfg):
    return cfg.constraint.purity_u1 - purity_c(dynamic_profile, cfg)




# ----------------------------------------------------------------------------- #
# ---------------------------- Tableting cont --------------------------------- #
# ----------------------------------------------------------------------------- #


# --- critical quality attribute constraints --- #
@partial(jit, static_argnums=(1))
def tablet_hardness(steady_state_outputs, cfg):
    # steady state outputs are in the order: [H, V_pre, V_main]
    return steady_state_outputs[0]


@partial(jit, static_argnums=(1))
def tablet_composition(steady_state_outputs, cfg):
    # steady state outputs are in the order: [hold up mass, hold_up volume, mass outflow rate, mass fraction of api, mass fraction of excipient, mass fraction of lubricant, porosity]
    return steady_state_outputs[3]


# ----- process constraints ----- #
@partial(jit, static_argnums=(1))
def tablet_size(steady_state_outputs, cfg):
    # steady state outputs are in the order: [H, V_pre, V_main]
    return steady_state_outputs[2] / (cfg.constraint.die_radius**2 * jnp.pi)


@partial(jit, static_argnums=(1))
def unit_volume(steady_state_outputs, cfg):
    # steady state outputs are in the order: hold up mass, hold_up volume ...
    return steady_state_outputs[1]


# --- constraint indicators --- #
@partial(jit, static_argnums=(1))
def tablet_hardness_lb(steady_state_outputs, cfg):
    return tablet_hardness(steady_state_outputs, cfg) - cfg.constraint.tablet_hardness[0]  # >= 0

@partial(jit, static_argnums=(1))
def tablet_hardness_ub(steady_state_outputs, cfg):
    return  cfg.constraint.tablet_hardness[1] - tablet_hardness(steady_state_outputs, cfg) # >= 0
        
@partial(jit, static_argnums=(1))
def tablet_composition_lb(steady_state_outputs, cfg):
    return tablet_composition(steady_state_outputs, cfg) - cfg.constraint.tablet_composition[0]  # >= 0

@partial(jit, static_argnums=(1))
def tablet_composition_ub(steady_state_outputs, cfg):
    return cfg.constraint.tablet_composition[1] - tablet_composition(steady_state_outputs, cfg) # >= 0

@partial(jit, static_argnums=(1))
def tablet_size_lb(steady_state_outputs, cfg):
    return tablet_size(steady_state_outputs, cfg) - cfg.constraint.tablet_size[0]

@partial(jit, static_argnums=(1))
def tablet_size_ub(steady_state_outputs, cfg):
    return cfg.constraint.tablet_size[1] - tablet_size(steady_state_outputs, cfg) 

@partial(jit, static_argnums=(1))
def unit1_volume_ub(steady_state_outputs, cfg):
    return cfg.constraint.unit1_volume - unit_volume(steady_state_outputs, cfg)

@partial(jit, static_argnums=(1))
def unit2_volume_ub(steady_state_outputs, cfg):
    return cfg.constraint.unit2_volume - unit_volume(steady_state_outputs, cfg)



# -------------------------------------------------------------------------------- #
# ---------------------------- batch_reaction_network ---------------------------- #
# -------------------------------------------------------------------------------- #


@partial(jit, static_argnums=(1))
def purity_b(dynamic_profile, cfg):
    pb = dynamic_profile[1] / jnp.sum(dynamic_profile[:])
    # jax.debug.print('pb {x}', x=pb)
    return pb


@partial(jit, static_argnums=(1))
def purity_c(dynamic_profile, cfg):
    return dynamic_profile[2] / jnp.sum(dynamic_profile[:])

@partial(jit, static_argnums=(1))
def purity_unit_2_brn_lb(dynamic_profile, cfg):
    return purity_b(dynamic_profile, cfg) - cfg.constraint.purity_u2


@partial(jit, static_argnums=(1))
def purity_unit_1_brn_ub(dynamic_profile, cfg):
    return cfg.constraint.purity_u1 - purity_c(dynamic_profile, cfg)



# -------------------------------------------------------------------------------- #
# ---------------------------- convex estimator --------------------------------- #
# -------------------------------------------------------------------------------- #

@partial(jit, static_argnums=(1))
def psd_constraint(dynamic_profile, cfg):
    return dynamic_profile[0] # this quantity must just be non-negative.

@partial(jit, static_argnums=(1))
def nonconvex_ground_truth(dynamic_profile, cfg):
    return non_convex_sum(dynamic_profile[-2:], cfg) + interaction_terms(dynamic_profile[-2:], cfg)

@partial(jit, static_argnums=(1))
def non_convex_sum( dynamic_profile, cfg):
    return jnp.sum(jnp.array([jnp.power(dynamic_profile[i],3) - jnp.power(dynamic_profile[i],2) for i in range(1,len(dynamic_profile))]))

@partial(jit, static_argnums=(1))
def interaction_terms(dynamic_profile, cfg):
    return dynamic_profile[0] * dynamic_profile[1]

@partial(jit, static_argnums=(1))
def estimation_bound_lb(dynamic_profile, cfg):
    return cfg.constraint.estimation_bound - jnp.linalg.norm(nonconvex_ground_truth(dynamic_profile, cfg) - dynamic_profile[0])



# -------------------------------------------------------------------------------- #
# --------------------- Affine constraints for the case study ---------------------#
# -------------------------------------------------------------------------------- #
@partial(jit, static_argnums=(1))
def negative_output_constraint(output, cfg):
    return output 


""" insert case study specific functions for constraints here"""
CS_holder = {'tablet_press': {0: [unit1_volume_ub], 1: [unit2_volume_ub, tablet_composition_lb, tablet_composition_ub], 2: [tablet_hardness_lb, tablet_hardness_ub, tablet_size_lb, tablet_size_ub]}, 
             'serial_mechanism_batch': {0: [purity_unit_1_ub], 1: [purity_unit_2_lb]},
             'convex_estimator': {0: [], 1: [], 2: [], 3: [], 4: [psd_constraint], 5: [estimation_bound_lb]},
             'affine_study': {0: [negative_output_constraint], 1: [negative_output_constraint], 2: [negative_output_constraint], 3: [negative_output_constraint], 4: [negative_output_constraint]},}
