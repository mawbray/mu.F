import networkx as nx
import numpy as np
import unittest
from graph.graph_assembly import graph_constructor
from cs_assembly import case_study_allocation

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
        G = case_study_allocation(G, cfg, {edge: 'edge_fn' for edge in G.G.edges}, constraint_dictionary, constraint_args)

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

if __name__ == '__main__':
    unittest.main()