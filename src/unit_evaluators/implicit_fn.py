
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



case_studies = {'tablet_press': {0: unit_1_dynamics, 1: unit_2_dynamics, 2: unit_3_dynamics}}