
from functools import partial 
from jax import jit
from omegaconf import DictConfig
import jax.numpy as jnp


@partial(jit, static_argnums=(0,))
def bulk_density_u1(cfg, design_args, input_args, *args):
    """bulk density function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        critical quality attributes (CQAs) - bulk density
        process constraints - None
        outputs - None

    """
    cfg_args = cfg.model.unit_1_args.bulk_density
    return (
        cfg_args[0]
        + cfg_args[1] * design_args[1]
        + cfg_args[2] * design_args[2]
        + cfg_args[3] * design_args[1] * design_args[1] * design_args[2]
    )


@partial(jit, static_argnums=(0,))
def mean_residence_time_u1(cfg, design_args, input_args, *args):
    """mean residence time function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        critical quality attributes (CQAs) - mean residence time
        process constraints - None
        outputs - None

    """
    cfg_args = cfg.model.unit_1_args.mean_residence_time
    cfg_d_args = cfg.case_study.KS_bounds[0][2]
    exp_term = -cfg_args[1] / (design_args[2] - cfg_d_args[0]) - cfg_args[2] / (
        design_args[0] + design_args[1]
    )
    return cfg_args[0] * (1 - jnp.exp(exp_term))


@partial(jit, static_argnums=(0,))
def unit_1_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 1 function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        outputs - hold up mass, hold_up volume, mass outflow rate, mass fraction of api, mass fraction of excipient

    """

    bulk_density = bulk_density_u1(cfg, design_args, input_args, *args) + design_args[-1] - design_args[-2] 
    tau_cm = mean_residence_time_u1(cfg, design_args, input_args, *args)
    hold_up = (design_args[0] + design_args[1]) * tau_cm

    return jnp.array(
        [
            hold_up,
            hold_up / bulk_density,
            design_args[0] + design_args[1],
            design_args[0] / (design_args[0] + design_args[1]),
            design_args[1] / (design_args[0] + design_args[1]),
        ]
    ).reshape(1, -1)


@partial(jit, static_argnums=(0,))
def hold_up_mass_u2(cfg, design_args, input_args, *args):
    """steady state hold up mass function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - steady state hold up mass

    """
    cfg_args = cfg.model.unit_2_args.hold_up_mass
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    return (
        cfg_args[0]
        + cfg_args[1] * mass_flow_in
        + cfg_args[2] * design_args[1]
        + cfg_args[3] * design_args[1] * design_args[1]
        + cfg_args[4] * mass_flow_in * design_args[1]
    )


@partial(jit, static_argnums=(0,))
def bulk_density_u2(cfg, design_args, input_args, *args):
    """bulk density function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - bulk density

    """
    cfg_args = cfg.model.unit_2_args.bulk_density
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    if not cfg.model.blender_density.include_lubricant:
        return (
            cfg_args[0]
            + cfg_args[1] * (input_args[1] / mass_flow_in) * design_args[1]
            + cfg_args[2] * (input_args[1] / mass_flow_in)
        )
    else:
        return (
            cfg_args[0]
            + cfg_args[1]
            * (input_args[1] / mass_flow_in)
            * design_args[1]
            * (design_args[0])
            / mass_flow_in
            + cfg_args[2] * (input_args[1] / mass_flow_in)
            + cfg_args[3] * design_args[0] / mass_flow_in
        )


@partial(jit, static_argnums=(0,))
def porosity_estimate_u2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """porosity estimate function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - porosity estimate

    """
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    # particle density calculation
    cfg_args = cfg.model.unit_2_args.particle_density
    if cfg.model.blender_density.include_lubricant:
        p_d = (
            cfg_args[0]
            + cfg_args[1]
            * design_args[1]
            * input_args[1]
            / (mass_flow_in)
            * design_args[0]
            / mass_flow_in
            + cfg_args[2] * input_args[1] / (mass_flow_in)
            + cfg_args[3] * design_args[0] / mass_flow_in
        )
    else:
        p_d = (
            cfg_args[0]
            + cfg_args[1] * design_args[1] * input_args[1] / (mass_flow_in)
            + cfg_args[2] * input_args[1] / (mass_flow_in)
        )
    # bulk density calculation
    p_bulk = bulk_density_u2(cfg, design_args, input_args, *args)
    return 1 - p_bulk / p_d, p_bulk


@partial(jit, static_argnums=(0,))
def unit_2_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 2 function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - hold up mass, hold_up volume, mass outflow rate, mass fraction of api, mass fraction of excipient, mass fraction of lubricant, porosity

    """
    input_args = input_args.squeeze()

    mass_hold_up = hold_up_mass_u2(cfg, design_args, input_args, *args) + design_args[-1] - design_args[-2] 
    porosity, bulk_density = porosity_estimate_u2(cfg, design_args, input_args, *args)
    mass_flow_out = input_args[0] + input_args[1] + design_args[0]

    return jnp.array(
        [
            mass_hold_up,
            mass_hold_up / bulk_density,
            mass_flow_out,
            input_args[0] / (mass_flow_out),
            input_args[1] / (mass_flow_out),
            design_args[0] / mass_flow_out,
            porosity,
        ]
    ).reshape(1, -1)


@partial(jit, static_argnums=(0,))
def main_comp_volume_unit_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """volume function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - pre-compression volume, pre-compression porosity

    Output:
        outputs - main-compression volume

    """
    V_pre, pre_comp_psty = args[0], args[1]
    main_comp_kawakita = cfg.model.unit_3_args.main_comp_kawakita
    numerator = V_pre * (1 - design_args[1] * main_comp_kawakita * (pre_comp_psty - 1))
    denominator = 1 + design_args[1] * main_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def pre_comp_volume_unit_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """pre comp volume function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - None

    Output:
        outputs - pre-compression volume

    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    pre_comp_kawakita = cfg.model.unit_3_args.pre_comp_kawakita
    numerator = V_0 * (1 - design_args[0] * pre_comp_kawakita * (input_args - 1))
    denominator = 1 + design_args[0] * pre_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def porosity_update_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """porosity update function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - pre-compression volume

    Output:
        outputs - porosity

    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    V_pre = args[0]
    return 1 - (1 - input_args) * V_0 / V_pre


def hardness_estimate_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """hardness estimate function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - main-compression volume

    Output:
        outputs - hardness

    """
    relative_density = (
        (1 - input_args) * cfg.model.unit_3_args.initial_volume_in_die / args[0]
    )
    gamma = jnp.log((1 - relative_density) / (1 - cfg.model.unit_3_args.critical_density))
    h_0 = cfg.model.unit_3_args.hardness_zero_porosity
    exp_term = relative_density - cfg.model.unit_3_args.critical_density + gamma
    return h_0 * (1 - jnp.exp(exp_term))


@partial(jit, static_argnums=(0,))
def unit_3_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 3 function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - porosity
    args - None

    Output:
        outputs - tablet hardness, pre-compression volume, main compression volume

    """
    input_args = input_args.squeeze() + design_args[-1] - design_args[-2] 
    V_pre = pre_comp_volume_unit_3(cfg, design_args, input_args, *args)
    porosity = porosity_update_u3(cfg, design_args, input_args, *(V_pre,))
    V_main = main_comp_volume_unit_3(cfg, design_args, input_args, *(V_pre, porosity))
    H = hardness_estimate_u3(cfg, design_args, input_args, *(V_main,))

    return jnp.array([H, V_pre, V_main]).reshape(1, -1)


import numpy as np
from pcgym import make_env
from functools import partial



def pcgym_fn_constructor(cfg, nodes):
    """
    Constructs a dictionary of functions that evaluate the constraints of the PCGym environment at each node of the graph.
    """
    config = cfg.model.pcgym
    T = config.T
    nsteps = config.nsteps 

    cons = {
        'T': config.cons.T
    } # list of constraints for each state variable
    
    cons_type = {
        'T': config.cons_type.T
    } # list of constraints type for each state variable
    
    SP = {
        'Ca': config.SP.Ca, # list of set points for each time step
    }

    r_scale = {
        'Ca': config.r_scale.Ca
    } # list of scales for each state variable

    action_space = {
        'low': np.array(config.action_space.low),
        'high':np.array(config.action_space.high)
    }

    #Continuous box observation space
    observation_space = {
        'low' : np.array(config.obs_space.low),
        'high' : np.array(config.obs_space.high),  
    }
    
    env_params = {
        'N': nsteps, # Number of time steps
        'tsim':T, # Simulation Time
        'SP':SP, # Setpoint
        'o_space' : observation_space, # Observation space
        'a_space' : action_space, # Action space
        'x0': np.array(config.x0), # Initial conditions 
        'model': str(config.model), # Select the model
        'r_scale': r_scale, # Scale the norm used for reward (|x-x_sp|*r_scale)
        'normalise_a': True, # Normalise the actions
        'normalise_o':True, # Normalise the states,
        'noise':False, # Add noise to the states
        'integration_method': 'casadi', # Select the integration method
        'noise_percentage':config.noise_percentage, # Noise percentage (scalar)
        'done_on_cons_vio':False,
        'constraints': cons, 
        'integration_method': str(config.integrator), # Select the integration method
        'cons_type': cons_type,
        'r_penalty': False
    }
    
    
    def fn_evaluator(cfg, params, collected_p, node, env):
        
        def evaluator(x, u, node, env):
            _ = env.reset()
            env.t = node
            if node > 0: # if not the initial state
                env.x = np.hstack([x.reshape(1,-1), np.array(env.SP['Ca'][node]).reshape(1,-1)]).reshape(-1,)
            x_, _, _, _, info = env.step(u)

            return np.expand_dims(np.hstack([x_.reshape(1,-1), - info["cons_info"][:, node, :].reshape(1,-1)]), axis=1) ## this works because the node is the time step and therefore the graph is serial in structure with all nodes having a single parent
        
        if params.ndim < 2:
            params = np.expand_dims(params, axis=1)
        u = params[:,:env.action_space.shape[0]]
        x = collected_p[:,:]
        return np.vstack([evaluator(x[i,:], u[i,:], node, env) for i in range(x.shape[0])])
    
    return {node: partial(fn_evaluator, node=node, env=make_env(env_params)) for node in range(nodes)}



case_studies = {'tablet_press': {0: unit_1_dynamics, 1: unit_2_dynamics, 2: unit_3_dynamics}, 'constrained_rl': pcgym_fn_constructor}