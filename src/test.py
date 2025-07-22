import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
import numpy as np
import jax.numpy as jnp
import unittest
from omegaconf import OmegaConf
from graph.graph_assembly import graph_constructor, build_graph_structure
from cs_assembly import case_study_allocation, case_study_constructor
from hydra import compose, initialize_config_dir

"""
Depcrecated (I believe)
class TestGraphAssembly(unittest.TestCase):
    def test_case_study_allocation(self):
        # Create a sample adjacency matrix
        adjacency_matrix = np.array([[0, 1, 0],
                                     [0, 0, 1],
                                     [0, 0, 0]])

        # Create a sample configuration
        class Config:
            class arrhenius:
                EA = {0: 1, 1: 2, 2: 3}
                R = 1
                A = {0: 4, 1: 5, 2: 6}
            n_design_args = {0: 2, 1: 1, 2: 1}
            n_input_args = {(0, 1): 1, (1, 2): 1}
            KS_bounds = {0: (0, 1), 1: (0, 1), 2: (0, 1)}
            parameters_best_estimate = {0: 0.5, 1: 0.8, 2: 0.2}
            parameters_samples = {0: [0.4, 0.6], 1: [0.7], 2: [0.1, 0.3]}
            fn_evals = {0: 10, 1: 20, 2: 30}
            unit_op = {0: 'dynamic', 1: 'dynamic', 2: 'dynamic'}
            unit_params_fn = {0: 'Arrhenius', 1: 'Arrhenius', 2: None}
            extendedDS_bounds = {0: (0, 1), 1: (0, 1), 2: (0, 1)}
            vmap_unit_evaluation = {0: False, 1: False, 2: False}
            arrhenius = arrhenius()

        cfg = Config()

        # Create a sample constraint dictionary
        constraint_dictionary = {0: 'constraint1', 1: 'constraint2', 2: 'constraint3'}

        # Create a sample constraint arguments dictionary
        constraint_args = {0: {'arg1': 1, 'arg2': 2}, 1: {'arg1': 3}, 2: {}}

        # Create a graph constructor object
        G = graph_constructor(cfg, adjacency_matrix)

        # Call the case_study_allocation function
        G = case_study_allocation(G, cfg, {edge: 'edge_fn' for edge in G.G.edges}, constraint_dictionary, constraint_args, {}, {})

        # Get the resulting graph
        graph = G.get_graph()

        # Assert the properties of the graph
        self.assertEqual(graph.nodes[0]['n_design_args'], 2)
        self.assertEqual(graph.nodes[1]['n_design_args'], 1)
        self.assertEqual(graph.nodes[2]['n_design_args'], 1)

        self.assertEqual(graph.nodes[0]['KS_bounds'], (0, 1))
        self.assertEqual(graph.nodes[1]['KS_bounds'], (0, 1))
        self.assertEqual(graph.nodes[2]['KS_bounds'], (0, 1))

        self.assertEqual(graph.nodes[0]['parameters_best_estimate'], 0.5)
        self.assertEqual(graph.nodes[1]['parameters_best_estimate'], 0.8)
        self.assertEqual(graph.nodes[2]['parameters_best_estimate'], 0.2)

        self.assertEqual(graph.nodes[0]['parameters_samples'], [0.4, 0.6])
        self.assertEqual(graph.nodes[1]['parameters_samples'], [0.7])
        self.assertEqual(graph.nodes[2]['parameters_samples'], [0.1, 0.3])

        self.assertEqual(graph.nodes[0]['fn_evals'], 10)
        self.assertEqual(graph.nodes[1]['fn_evals'], 20)
        self.assertEqual(graph.nodes[2]['fn_evals'], 30)

        self.assertEqual(graph.nodes[0]['unit_op'], 'dynamic')
        self.assertEqual(graph.nodes[1]['unit_op'], 'dynamic')
        self.assertEqual(graph.nodes[2]['unit_op'], 'dynamic')

        self.assertEqual(graph.nodes[0]['unit_params_fn'], 'Arrhenius')
        self.assertEqual(graph.nodes[1]['unit_params_fn'], 'Arrhenius')
        self.assertEqual(graph.nodes[2]['unit_params_fn'], None)

        self.assertEqual(graph.nodes[0]['extendedDS_bounds'], (0, 1))
        self.assertEqual(graph.nodes[1]['extendedDS_bounds'], (0, 1))
        self.assertEqual(graph.nodes[2]['extendedDS_bounds'], (0, 1))

        self.assertEqual(graph.nodes[0]['constraints'], 'constraint1')
        self.assertEqual(graph.nodes[1]['constraints'], 'constraint2')
        self.assertEqual(graph.nodes[2]['constraints'], 'constraint3')

        self.assertEqual(graph.nodes[0]['constraint_args'], {'arg1': 1, 'arg2': 2})
        self.assertEqual(graph.nodes[1]['constraint_args'], {'arg1': 3})
        self.assertEqual(graph.nodes[2]['constraint_args'], {})

        self.assertEqual(graph.edges[(0, 1)]['n_input_args'], 1)
        self.assertEqual(graph.edges[(1, 2)]['n_input_args'], 1)

        self.assertEqual(graph.edges[(0, 1)]['input_indices'], [1])
        self.assertEqual(graph.edges[(1, 2)]['input_indices'], [1])

        """

class TestTemporalStudyRewards(unittest.TestCase):
    def setUp(self):
        # Initialize hydra config for temporal study
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        with initialize_config_dir(config_dir=config_path, version_base=None):
            self.cfg = compose(config_name='integrator.yaml', 
                              overrides=['case_study=temporal_study',
                                         'model=temporal_study',
                                       'case_study.number_repeats=3',
                                       'case_study.eval_rewards=true'])
    
    def test_graph_reward_integration(self):
        # Build graph structure for serial temporal study
        cfg = build_graph_structure(self.cfg)
        
        # Construct case study graph
        G = case_study_constructor(cfg)
        
        # Verify graph has correct serial structure
        nodes = list(G.nodes)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(list(nx.topological_sort(G)), [0, 1, 2])
        
        # Test reward computation at each node
        test_design = jnp.array([[2.0, 3.0]])
        test_input = jnp.array([[5.0]])  # Initial inlet flow
        
        rewards = []
        outputs = []
        
        # Process each node in forward order
        for node in nodes:
            unit_eval = G.nodes[node]['forward_evaluator']
            
            # Compute output and reward
            output = unit_eval.evaluate(test_design, test_input, aux_args=jnp.array([[]]), uncertain_params=jnp.array([[0.]]))
            reward = unit_eval.rewards(test_design, test_input, aux_args=jnp.array([[]]), uncertain_params=jnp.array([[0.]]))
            
            rewards.append(reward)
            outputs.append(output)
            
            # Output becomes input for next node
            if node < len(nodes) - 1:
                test_input = output
        
        # Verify rewards were computed
        self.assertEqual(len(rewards), 3)
        for i, reward in enumerate(rewards):
            self.assertIsNotNone(reward.squeeze(), f"Node {i} reward should not be None")
        
        # Verify rewards change between nodes due to flow coupling
        self.assertFalse(jnp.allclose(rewards[0], rewards[1]))
        
        # Verify all nodes have reward functions
        for node in nodes:
            self.assertTrue(hasattr(G.nodes[node]['forward_evaluator'].unit_cfg, 'reward_fn'))
    
    def test_reward_calculation_correctness(self):
        # Simple test with known values
        cfg = build_graph_structure(self.cfg)
        cfg.model.coeff.A = [1.0, 2.0, 3.0]
        cfg.model.coeff.B = [0.5, 1.0, 0.0]
        
        G = case_study_constructor(cfg)
        unit_eval = G.nodes[0]['forward_evaluator']
        
        # Test with simple inputs
        design = jnp.array([[2.0, 3.0]])
        inlet = jnp.array([[5.0]])
        
        reward = unit_eval.rewards(design, inlet, aux_args=jnp.array([[]]), uncertain_params=jnp.array([[0., 0.]]))
        
        # Expected: A·x - inlet*B·x where x = [2, 3, 1]
        # A·x = 1*2 + 2*3 + 3*1 = 11
        # B·x = 0.5*2 + 1*3 + 0*1 = 4
        # reward = 11 - 5*4 = -9
        expected = -9.0
        self.assertAlmostEqual(float(reward), expected, places=5)
    
    def test_complete_graph_forward_backward_consistency(self):
        # Test that forward flow and rewards are consistent through the entire graph
        cfg = build_graph_structure(self.cfg)
        G = case_study_constructor(cfg)
        
        # Multiple design points to test
        designs = [jnp.array([[1.0, 2.0]]), jnp.array([[3.0, 4.0]]), jnp.array([[0.5, 1.5]])]
        inlets = [jnp.array([[5.0]]), jnp.array([[10.0]]), jnp.array([[2.0]])]
        
        for design, inlet in zip(designs, inlets):
            outputs = []
            rewards = []
            current_input = inlet
            
            # Forward pass through graph
            for node in [0, 1, 2]:
                unit_eval = G.nodes[node]['forward_evaluator']
                output = unit_eval.evaluate(design, current_input, 
                                          aux_args=jnp.array([[]]), 
                                          uncertain_params=jnp.array([[0.]]))
                reward = unit_eval.rewards(design, current_input,
                                         aux_args=jnp.array([[]]),
                                         uncertain_params=jnp.array([[0.]]))
                outputs.append(output)
                rewards.append(float(reward.squeeze()))
                current_input = output
            
            # Verify outputs flow correctly (each output becomes next input)
            self.assertEqual(outputs[0].shape, inlet.shape)
            self.assertEqual(outputs[1].shape, outputs[0].shape)
            self.assertEqual(outputs[2].shape, outputs[1].shape)
            
            # Verify rewards change through the graph due to flow coupling
            self.assertNotEqual(rewards[0], rewards[1])
            self.assertNotEqual(rewards[1], rewards[2])
            
            # Verify reward magnitudes are reasonable (not NaN or Inf)
            for r in rewards:
                self.assertFalse(jnp.isnan(r))
                self.assertFalse(jnp.isinf(r))

if __name__ == '__main__':
    unittest.main()