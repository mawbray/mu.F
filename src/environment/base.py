"""
Base environment configuration - all environments should inherit from this.

NOTE:
    - This environment will be used for both dynamic programming evalutation and testing. 
    - The step interfact will be used in simulation, whilst F and G will be used in Mu_F.
"""


import jax.numpy as jnp

from omegaconf import OmegaConf, DictConfig
from abc import ABC, abstractmethod
from operator import ge, le

from agent.base import HYDRA_CONFIG_FILE, OUTPUTS_DIR

class DeterministicNode(ABC):
    U_SIZE = None  # Size of input observation vector
    V_SIZE = None  # Size of action vector
    Y_SIZE = None  # Size of output observation vector
    X_SIZE = None  # Size of constraint space vector
    """
    Here we will use mu-F nomenclature for states and actions.

    Some notes:
        - u are observations.
        - v are actions.
        - y = F(u,v) are ouputs.
        - x = G(u,v) are constraint spaces

    """
    def __init__(self, **kwargs):
        self.cfg = self._build_cfg(**kwargs)
        self._set_infeas_sign()
        self.model_cfg = self.cfg.model if hasattr(self.cfg, "model") else self.cfg
        self.current_step = 0
        self.max_steps = self.model_cfg.number_repeats
        self._cache = self.model_cfg.memory
        self._initialise_env(self.cfg)

    # ---- Class methods ---- #
    @classmethod
    def F(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., :cls.Y_SIZE]

    @classmethod
    def G(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., cls.Y_SIZE:cls.Y_SIZE + cls.X_SIZE]

    @classmethod
    def R(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., cls.Y_SIZE + cls.X_SIZE:]

    # ---- Abstract methods ---- #

    def _initialise_env(self, cfg):
        """Initialise the environment - to be implemented in derived classes"""
        model_cfg = cfg.model if hasattr(cfg, "model") else cfg
        self._feas_thresh = model_cfg.feas_thresh
        return model_cfg

    @abstractmethod
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    # ---- Shared methods ---- #
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self._initialise_env(self.cfg)
        return jnp.array(self.cfg.model.root_node_inputs)

    def step(self, u, v):
        """
        Step method to take action v given observation u.

        Parameters:
            - u : observations
            - v : actions

        Returns:
            - y : outputs
            - x : constraint spaces
        """
        
        
        output = self.simulate(u, v)

        y = self.F(output)
        x = self.G(output)
        reward = self.R(output)
        
        term, trunc = self._termination_conditions(x)
        self._tick()
    
        if term and not trunc: reward = 1000

        return  y, reward, term, trunc, {'constraint_values': x}
    
    def _termination_conditions(self, x):
        """Termination conditions for the environment"""
        test = [self._infeas_sign(x_i, self._feas_thresh) for x_i in x]
        if jnp.any(jnp.array(test)):
            term = True
            trunc = False
        elif self.current_step >= self.max_steps:
            term = True
            trunc = True
        else:
            term = False
            trunc = False

        return term, trunc

    def _tick(self):
        """Increment the current step"""
        self.current_step += 1
        return self.current_step
    
    def _build_cfg(self, **kwargs):
        """Build the configuration for the environment"""
        if 'cfg' in kwargs:
            if isinstance(kwargs['cfg'], dict):
                cfg = OmegaConf.create(kwargs['cfg'])
            elif(isinstance(kwargs['cfg'], DictConfig)):
                cfg = kwargs['cfg']
            else:
                pass
        elif 'solve_date' in kwargs and 'solve_id' in kwargs:
            cfg = OmegaConf.load(OUTPUTS_DIR.format(solve_date=kwargs['solve_date'], solve_id=kwargs['solve_id']) + HYDRA_CONFIG_FILE)
        else:
            raise ValueError('No configuration provided for environment')
        return cfg

    def _set_infeas_sign(self):
        """Sets the notion of infeasibility"""
        if hasattr(self.cfg, 'samplers'):
            self._infeas_sign = le if self.cfg.samplers.notion_of_feasibility == 'positive' else ge
        else:
            self._infeas_sign = None
        
