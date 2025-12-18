from enum import Enum
from typing import Optional, Callable, Any, Tuple
from omegaconf import DictConfig
from mu_F.solvers.solvers import serialms_casadi_box_eq_nlp_solver, jax_box_nlp_solver, parallel_casadi_box_eq_nlp_solver


class SolverType(str, Enum):
    """Enumeration of supported solver types."""
    GENERAL_CONSTRAINED_NLP = "general_constrained_nlp"
    BOX_CONSTRAINED_NLP = "box_constrained_nlp"


def validate_solver_config(cfg: DictConfig) -> None:
    """
    Validate solver configuration.
    
    Checks that required solver configuration keys exist and that solver types
    are valid.
    
    Args:
        cfg: Configuration object to validate
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    required_keys = [
        'forward_coupling_solver',
        'backward_coupling_solver',
        'forward_coupling',
        'backward_coupling'
    ]
    
    # Check for required keys in cfg.solvers
    if not hasattr(cfg, 'solvers'):
        raise ValueError("Missing 'solvers' section in configuration")
    
    for key in required_keys:
        if not hasattr(cfg.solvers, key):
            raise ValueError(f"Missing required solver config: solvers.{key}")
    
    # Validate solver types
    try:
        SolverType(cfg.solvers.forward_coupling_solver)
    except ValueError as e:
        raise ValueError(
            f"Invalid forward solver type '{cfg.solvers.forward_coupling_solver}'. "
            f"Supported types: {[t.value for t in SolverType]}"
        ) from e
    
    try:
        SolverType(cfg.solvers.backward_coupling_solver)
    except ValueError as e:
        raise ValueError(
            f"Invalid backward solver type '{cfg.solvers.backward_coupling_solver}'. "
            f"Supported types: {[t.value for t in SolverType]}"
        ) from e


class solver_construction:
    """
    Base class used to construct local solvers for the feasibility problem.
    
    This class provides a factory pattern for creating configured solver instances.
    Use from_method() to create fully configured solvers.
    """
    
    def __init__(self, cfg: DictConfig, solver_type: SolverType):
        """
        Initialize solver construction.
        
        Args:
            cfg: Configuration object containing solver settings
            solver_type: Type of solver to construct
        """
        self.cfg = cfg
        self.solver_type = solver_type
        self.objective_func: Optional[Callable] = None
        self.constraints_func: Optional[Callable] = None
        self.bounds: Optional[Tuple] = None
        self.solver: Optional[Any] = None

    @classmethod
    def from_method(
        cls,
        cfg: DictConfig,
        solver_type: str,
        objective_func: Callable,
        bounds: Tuple,
        eq_constraints_func: Optional[Callable] = None
    ) -> 'solver_construction':
        """
        Factory method to create a fully configured solver instance.
        
        This is the recommended way to create solver instances. It creates a new
        instance, loads the objective function, constraints, and bounds, then
        constructs the underlying solver.
        
        Args:
            cfg: Configuration object containing solver settings
            solver_type: String identifier for solver type (will be converted to SolverType)
            objective_func: Objective function for the solver
            bounds: Tuple of (lower_bounds, upper_bounds)
            eq_constraints_func: Optional equality constraints function
            
        Returns:
            Fully configured solver_construction instance
            
        Raises:
            ValueError: If objective_func is None
            ValueError: If solver_type is invalid
        """
        # Convert string to SolverType enum for validation
        try:
            solver_type_enum = SolverType(solver_type)
        except ValueError:
            raise ValueError(
                f"Unknown solver type '{solver_type}'. "
                f"Supported types: {[t.value for t in SolverType]}"
            )
        
        new_solver_class = cls(cfg, solver_type_enum)
        
        if objective_func is None:
            raise ValueError('Objective function must be provided')
        
        new_solver_class.load_equality_constraints(eq_constraints_func)
        new_solver_class.load_objective(objective_func)
        new_solver_class.load_bounds(bounds)
        new_solver_class.construct_solver()
        return new_solver_class

    def __call__(self, initial_guesses: Any) -> Any:
        """
        Solve the optimization problem with given initial guesses.
        
        Args:
            initial_guesses: Initial guess values for the solver
            
        Returns:
            Solver results (format depends on solver type)
        """
        if self.solver_type == SolverType.GENERAL_CONSTRAINED_NLP:
            return self.solver(initial_guesses)
        else:
            return self.solver.solve(initial_guesses)
    
    def construct_solver(self) -> None:
        """
        Construct the actual solver instance based on solver type.
        
        Raises:
            NotImplementedError: If solver type is not implemented
        """
        if self.solver_type == SolverType.GENERAL_CONSTRAINED_NLP:
            if self.cfg.parallelised:
                self.solver = parallel_casadi_box_eq_nlp_solver(
                    self.cfg, self.objective_func, self.constraints_func, self.bounds
                )
            else:
                self.solver = serialms_casadi_box_eq_nlp_solver(
                    self.cfg, self.objective_func, self.constraints_func, self.bounds
                )
        elif self.solver_type == SolverType.BOX_CONSTRAINED_NLP:
            self.solver = jax_box_nlp_solver(self.cfg, self.objective_func, self.bounds)
        else:
            raise NotImplementedError(
                f'Solver type {self.solver_type.value} not implemented. '
                f'Supported types: {[t.value for t in SolverType]}'
            )
        
    def load_objective(self, objective_func: Callable) -> None:
        """Load the objective function."""
        self.objective_func = objective_func
    
    def load_equality_constraints(self, constraints_func: Optional[Callable]) -> None:
        """Load the equality constraints function."""
        self.constraints_func = constraints_func
    
    def load_bounds(self, bounds: Tuple) -> None:
        """Load the variable bounds."""
        self.bounds = bounds
        
    def initial_guess(self) -> Any:
        """
        Generate initial guess for the solver.
        
        Returns:
            Initial guess values (format depends on solver type)
            
        Raises:
            NotImplementedError: If solver type is not implemented
        """
        if self.solver_type == SolverType.GENERAL_CONSTRAINED_NLP:
            return self.solver.initial_guess()
        elif self.solver_type == SolverType.BOX_CONSTRAINED_NLP:
            return self.solver.initial_guess()
        else:
            raise NotImplementedError(
                f'Solver type {self.solver_type.value} not implemented. '
                f'Supported types: {[t.value for t in SolverType]}'
            )