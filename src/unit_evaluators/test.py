import unittest
from dataclasses import dataclass, field
import jax.numpy as jnp
from constructor import unit_evaluation

@dataclass(frozen=False, kw_only=True)
class arrhenius:
    EA: list
    A: list
    R: float

    def __init__(self):
        self.EA = [2500.2, 5000.1]
        self.A = [6.66e-6, 0.0010335]
        self.R = 8.314

@dataclass(frozen=False, kw_only=True)
class integration:
    scheme: dict
    step_size_controller: str
    max_steps: int
    t0: float
    tf: float
    dt0: float

    def __init__(self):
        self.scheme = {0: 'tsit5', 1: 'tsit5'}
        self.step_size_controller = 'pid'
        self.max_steps = 10000
        self.t0 = 0.0
        self.tf = 1.0
        self.dt0 = 0.01

@dataclass(frozen=False, kw_only=False)
class config:
    study: str
    n_units: int
    include_intermediate_constraint: bool
    vmap_unit_evaluation: bool
    case_study_dynamics: str
    arrhenius: object=field(default_factory=lambda :arrhenius()) 
    units: list=field(default_factory= lambda: [0, 1])
    root_node_inputs: list=field(default_factory= lambda: [[2000., 0., 0.]])
    integration: object=field(default_factory=lambda :integration()) 

    def __init__(self):
        self.study = 'study_0'
        self.n_units = 2
        self.include_intermediate_constraint = False
        self.vmap_unit_evaluation = [True, True]
        self.case_study_dynamics ='serial_batch'
        self.integration= integration()
        self.arrhenius= arrhenius()
        #self.units = units()
    
    def __post__init__(self):
        self.units = [0, 1]
        self.root_node_inputs = [[2000., 0., 0.]]*2


@dataclass(frozen=False, kw_only=True)
class graph_obj:
    nodes: dict=field(default_factory= lambda: {0: {'unit_op': 'dynamic', 'unit_params_fn': 'Arrhenius', 'unit_uncertainty': True}, 
    1: {'unit_op': 'dynamic', 'unit_params_fn': 'Arrhenius', 'unit_uncertainty': True}})


class TestUnitEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a sample configuration object, graph, and node
        cfg = config()  # Replace with your actual configuration object
        cfg.__post__init__()
        graph = graph_obj()  # Replace with your actual graph object
        node = 1  # Replace with your actual node
        list_ = cfg.arrhenius
        # Create a unit_evaluation object for testing
        self.unit_eval = unit_evaluation(cfg, graph, node)

    def test_evaluate(self):
        # Define sample design arguments and input arguments
        design_args = jnp.array([1, 2, 3]*2).reshape(2,-1)  # Replace with your actual design arguments

        input_args = jnp.array([4, 5, 6]*2).reshape(2,-1)  # Replace with your actual input arguments

        uncertain_params = jnp.array([7, 8, 9]*5).reshape(5,-1)

        # Call the evaluate method and get the result
        result = self.unit_eval.evaluate(design_args, None, uncertain_params)


        # Assert that the result is as expected
        if self.unit_eval.cfg.vmap_unit_evaluation:
            expected_result = (design_args.shape[0],uncertain_params.shape[0], 3)  # Replace with your expected result
        else: 
            expected_result = (3,)
        self.assertEqual(result.shape, expected_result)

if __name__ == '__main__':
    unittest.main()