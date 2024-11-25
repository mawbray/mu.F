import numpy as np
from pcgym import make_env
from functools import partial




def pcgym_fn_constructor(cfg, G):


    config = cfg.model.pcgym
    T = config.T
    nsteps = config.nsteps 
    cons = config.cons
    cons_type = config.cons_type
    
    SP = {
        'Ca': config.SP, # list of set points for each time step
    }

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
        'x0': np.array([config.x0]), # Initial conditions 
        'model': config.model, # Select the model
        'normalise_a': True, # Normalise the actions
        'normalise_o':True, # Normalise the states,
        'noise':True, # Add noise to the states
        'integration_method': 'casadi', # Select the integration method
        'noise_percentage':config.noise_percentage, # Noise percentage (scalar)
        'done_on_cons_vio':False,
        'constraints': cons, 
        'cons_type': cons_type,
        'r_penalty': True
    }
    
    
    def fn_evaluator(x, u, node, env):
        x = env.reset()
        env.t = node
        env.x = x
        _, _, _, info = env.step(u)
        return info["cons_info"][:, node, :]
    
    

    return {node: partial(fn_evaluator, node=node, env=make_env(env_params)) for node in G.nodes}
