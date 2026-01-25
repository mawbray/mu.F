"""
Base environment configuration - all environments should inherit from this.

NOTE:
    - This environment will be used for both dynamic programming evalutation and testing. 
    - The step interfact will be used in simulation, whilst F and G will be used in Mu_F.
"""

import weakref
import jax.numpy as jnp

from abc import ABC, abstractmethod

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg = cfg.model if hasattr(cfg, "model") else cfg
        self.current_step = 0
        self.max_steps = self.model_cfg.number_repeats
        self._cache = self.model_cfg.memory
        self._initialise_env(cfg)

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

    @abstractmethod
    def _initialise_env(self, cfg):
        """Initialise the environment - to be implemented in derived classes"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    # ---- Shared methods ---- #
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self._initialise_env(self.cfg)
        pass

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
        u_v = jnp.concatenate([u, v], axis=-1)
        self._tick()

        output = self.simulate(u_v)

        y = self.F(output)
        x = self.G(output)
        reward = self.R(output)
        term, trunc = self._termination_conditions(x)
        if term and not trunc: reward = self.model_cfg.infeasibility_penalty

        return  y, reward, term, trunc, {'constraint_values': x}
    
    def _termination_conditions(self, x):
        """Termination conditions for the environment"""
        if jnp.any(x < self.model_cfg.feas_thresh):
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
    
