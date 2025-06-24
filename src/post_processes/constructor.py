from abc import ABC
import jax.numpy as jnp
import numpy as np
from jax.random import choice, PRNGKey


class post_process_base(ABC):
    def __init__(self, cfg, graph, model):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        pass

    def run(self):
        pass

    def load_feasible_infeasible(self, feasible, live_set):
        self.feasible = feasible
        self.live_set = live_set

    def load_training_methods(self, training_methods):
        assert hasattr(training_methods, 'train')
        self.training_methods = training_methods

    def load_solver_methods(self, solver_methods):
        self.solver_methods = solver_methods

    def construct_descriptive_graph(self,):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def solve_for_nuisnace_parameters(self, decision_variables: list[int], bounds: list[list[float]], notion: str= 'max') -> list:
        """
        Solve for nuisance parameters
        :param decision_variables: The decision variables
        :param bounds: The bounds for the decision variables
        :param notion: The notion of the problem (max or min)
        :return: The solution for the having factored out nuisance parameters
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


class post_process(post_process_base):
    def __init__(self, cfg, graph, model):
        super().__init__(cfg, graph, model)
        self.feasible = None
        self.live_set = None
        self.training_methods = None
        self.solver_methods = None

    def run(self):
        # Implement the main logic for post-processing here
        pass

    def construct_descriptive_graph(self):
        # Implement the logic to construct a descriptive graph
        pass

    def solve_for_nuisance_parameters(self, decision_variables: list[int], bounds: list[list[float]], notion: str= 'max') -> list:
        # Implement the logic to solve for nuisance parameters
        pass

    def load_trainer_methods(self, trainer_methods):
        """
        Load the trainer methods for the post-processing
        :param trainer_methods: The trainer methods to be loaded
        """
        assert hasattr(trainer_methods, 'train')
        self.trainer_methods = trainer_methods