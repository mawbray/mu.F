
from functools import lru_cache, partial
from environment.base import DeterministicNode

import jax.numpy as jnp
import jax



class H2ExportEnvironment(DeterministicNode):
    U_SIZE = 2  # [_hydrogen_storage, _vector_throughput]
    V_SIZE = 1  # [vector_throughput]
    Y_SIZE = 2  # [ hydrogen_storage, vector_throughput]
    X_SIZE = 4  # [lower_ramp_limit, upper_ramp_limit, lower_h2_storage, upper_h2_storage]
    """
    H2 Export Environment

    Notation:
        - u : [_hydrogen_storage, _vector_throughput]
        - v : [vector_throughput]
        - z : [_renewable_energy]
        - y : [renewable_energy, hydrogen_storage, vector_throughput]
        - x : [lower_ramp_limit, upper_ramp_limit, lower_h2_storage, \
                upper_h2_storage]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ---- Interface with Mu.F ---- #
    def __call__(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        assert u.shape[-1] == self.U_SIZE, f"Expected last dimension {self.U_SIZE}, got {u.shape[-1]}"
        assert v.shape[-1] == self.V_SIZE, f"Expected last dimension {self.V_SIZE}, got {v.shape[-1]}"
        return self.simulate(
            u,
            v,
            z=self._renewable_energy_value * jnp.ones_like(u[..., :1]),
        )
    
    @partial(jax.jit, static_argnums=0)
    def simulate(self, u: jnp.ndarray, v: jnp.ndarray, z: jnp.ndarray = None) -> jnp.ndarray:
        """
        Simulation step for the H2 Export model.
        """
        # Implement the simulation logic
        _hydrogen_storage = u[..., 0]
        _vector_throughput = u[..., 1]
        _renewable_energy = (
            z[..., 0] if z is not None else self._renewable_energy_value
        )
        vector_throughput = v[..., 0]

        # Simulate the model dynamics here
        _active_trains = number_active_trains_eq(
            _vector_throughput,
            self._train_throughput_capacity,
            self._vector_calorific_value,
        )
        active_trains = number_active_trains_eq(
            vector_throughput,
            self._train_throughput_capacity,
            self._vector_calorific_value,
        )
        vector_energy = vector_energy_eq(
            vector_throughput,
            active_trains,
            self._variable_energy_penalty,
            self._vector_calorific_value,
            self._fixed_energy_penalty,
            self._train_throughput_capacity,
        )
        energy_electrolysis = energy_electrolysis_eq(_renewable_energy, vector_energy, self._n_turbines)
        energy_fuelcell = energy_fuelcell_eq(_renewable_energy, vector_energy, self._n_turbines)
        hydrogen_delta = hydrogen_delta_eq(
            vector_throughput,
            energy_electrolysis,
            energy_fuelcell,
            self._vector_molar_efficiency,
            self._electrolyser_efficiency,
            self._fuelcell_efficiency,
        )
        hydrogen_storage = hydrogen_storage_eq(_hydrogen_storage, hydrogen_delta)


        # Calculate constraints
        lower_ramp_cons = vector_ramping_lower_cons(
            vector_throughput,
            _vector_throughput,
            _active_trains,
            self._vector_calorific_value,
            self._lower_ramp_limit,
            self._train_throughput_capacity,
        )
        upper_ramp_cons = vector_ramping_upper_cons(
            vector_throughput,
            _vector_throughput,
            _active_trains,
            self._vector_calorific_value,
            self._upper_ramp_limit,
            self._train_throughput_capacity,
            self._n_trains_conversion,
        )
        lower_h2_storage_cons = hydrogen_storage_lower_cons(
            hydrogen_storage,
            self._lower_storage_limit,
            self._upper_storage_limit,
        )
        upper_h2_storage_cons = hydrogen_storage_upper_cons(hydrogen_storage, self._upper_storage_limit)

        # Calculate reward
        reward = jnp.broadcast_to(-vector_throughput, hydrogen_storage.shape)

        # Stack outputs and constraints
        outputs = jnp.stack([hydrogen_storage, vector_throughput], axis=-1)
        constraints = jnp.stack([lower_ramp_cons, upper_ramp_cons, lower_h2_storage_cons, upper_h2_storage_cons], axis=-1)
        reward = jnp.expand_dims(reward, axis=-1)

        combined = jnp.concatenate([outputs, constraints, reward], axis=-1)
        assert combined.shape[-1] == self.Y_SIZE + self.X_SIZE + 1, f"Expected last dimension {self.Y_SIZE + self.X_SIZE + 1}, got {combined.shape[-1]}"

        if self._cache:
            self._constraints = constraints
            self._outputs = outputs
            self._reward = reward
            
        return combined
    
    
    def _initialise_env(self, cfg):
        """
        Initialise the H2 Export model.
        """
        model_cfg = super()._initialise_env(cfg)
        self._hydrogen_storage_cap = model_cfg.hydrogen_storage_capacity
        self._vector_storage_cap = model_cfg.vector_storage_capacity
        self._n_turbines = model_cfg.number_of_turbines
        self._n_trains_conversion = model_cfg.number_of_trains_conversion
        self._hfc_cap = model_cfg.hfc_capacity
        self._renewable_energy_value = model_cfg.renewable_energy_value
        self._train_throughput_capacity = model_cfg.constants.train_throughput_capacity
        self._vector_molar_efficiency = model_cfg.constants.vector_molar_efficiency
        self._electrolyser_efficiency = model_cfg.constants.electrolyser_efficiency
        self._fuelcell_efficiency = model_cfg.constants.fuelcell_efficiency
        self._fixed_energy_penalty = model_cfg.constants.fixed_energy_penalty
        self._variable_energy_penalty = model_cfg.constants.variable_energy_penalty
        self._vector_calorific_value = model_cfg.constants.vector_calorific_value
        self._lower_ramp_limit = model_cfg.constants.lower_ramp_limit
        self._upper_ramp_limit = model_cfg.constants.upper_ramp_limit
        self._lower_storage_limit = model_cfg.constants.lower_storage_limit
        self._upper_storage_limit = model_cfg.constants.upper_storage_limit



# -------------------------------------------------------------------------------- #
# ------------------------------ Equations --------------------------------------- #
# -------------------------------------------------------------------------------- #
@jax.jit
def number_active_trains_eq(vector_throughput, train_throughput_capacity, vector_calorific_value):
    """ Calculate the number of active trains based on vector throughput """
    # (GJ/ h) / (t(NH3) / train * GJ/t(NH3)) = trains
    return jnp.ceil(vector_throughput / (train_throughput_capacity * vector_calorific_value))

@jax.jit
def energy_electrolysis_eq(renewable_energy,  vector_energy, num_turbines):
    """ Calculate the energy used for electrolysis """
    # Number * GJ / h - GJ / h = GJ / h
    return jnp.maximum(renewable_energy * num_turbines - vector_energy, 0)

@jax.jit
def energy_fuelcell_eq(renewable_energy,  vector_energy, num_turbines):
    """ Calculate the energy used by the fuelcell """
    # GJ / h - GJ / h = GJ / h
    return jnp.maximum(vector_energy - renewable_energy * num_turbines, 0)

@jax.jit
def hydrogen_storage_eq(hydrogen_storage_prev, hydrogen_delta):
    """ Update hydrogen storage based on stored hydrogen and throughput """
    # GJ + GJ = GJ
    return hydrogen_storage_prev + hydrogen_delta

@jax.jit
def hydrogen_delta_eq(vector_throughput, energy_electrolysis, energy_fuelcell, vector_molar_efficiency, electrolyser_efficiency, fuelcell_efficiency):
    """ Calculate hydrogen removed based on vector throughput """
    # (GJ / h) / (-) - - (GJ / h) / (-) - (GJ / h) / (-) = GJ / h
    return vector_throughput / vector_molar_efficiency - energy_electrolysis / electrolyser_efficiency - energy_fuelcell / fuelcell_efficiency

@jax.jit
def vector_energy_eq(vector_throughput, number_active_trains, variable_energy_penalty, vector_calorific_value, fixed_energy_penalty, train_throughput_capacity):
    """ Update vector energy based on previous energy and throughput """
    # GJ / h * ( GJ / tonne (NH3) * (tonne(NH3) / GJ)) * (1 - (-)) + Number * (-) * GJ / tonne (NH3) * (tonne(NH3) / h) = GJ / h
    return (
        vector_throughput * (variable_energy_penalty / vector_calorific_value) * (1 - fixed_energy_penalty) 
        + number_active_trains * fixed_energy_penalty * variable_energy_penalty * train_throughput_capacity 
    )

# -------------------------------------------------------------------------------- #
# ------------------------------ Constraints ------------------------------------- #
# -------------------------------------------------------------------------------- #
@jax.jit    
def vector_ramping_lower_cons(vector_throughput, _vector_throughput, _active_trains, vector_calorific_value, lower_ramp_limit, train_throughput_capacity): 
    """ Constraint for lower ramping limit of vector energy """
    # GJ(NH3) / h - GJ(NH3) / h - (-) * Number * GJ(NH3) / h = GJ(NH3) / h
    return - (
        (_vector_throughput - vector_throughput) / vector_calorific_value
        - lower_ramp_limit * (_active_trains) * train_throughput_capacity
    ) / (lower_ramp_limit * (_active_trains) * train_throughput_capacity)

@jax.jit
def vector_ramping_upper_cons(vector_throughput, _vector_throughput, _active_trains, vector_calorific_value, upper_ramp_limit, train_throughput_capacity, total_trains): 
    """ Constraint for upper ramping limit of vector energy """
    # GJ(NH3) / h - GJ(NH3) / h - (-) * Number * GJ(NH3) / h = GJ(NH3) / h
    return - (
        (vector_throughput - _vector_throughput) / vector_calorific_value
        - upper_ramp_limit * (total_trains - _active_trains + 1) * train_throughput_capacity
    ) / (upper_ramp_limit * (total_trains - _active_trains + 1) * train_throughput_capacity)

@jax.jit
def hydrogen_storage_lower_cons(hydrogen_storage, lower_storage_limit, upper_storage_limit):
    """ Constraint for lower hydrogen storage limit """
    # GJ - (-) * GJ = GJ
    return (hydrogen_storage - lower_storage_limit * upper_storage_limit) / upper_storage_limit

@jax.jit
def hydrogen_storage_upper_cons(hydrogen_storage, upper_storage_limit):
    """ Constraint for upper hydrogen storage limit """
    # GJ - GJ = GJ
    return (upper_storage_limit - hydrogen_storage) / upper_storage_limit
