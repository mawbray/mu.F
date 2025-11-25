import pytest
import networkx as nx
import numpy as np
from unittest.mock import MagicMock, patch
from mu_F.direct import apply_direct_method

@patch("mu_F.direct.construct_deus_problem_network")
@patch("mu_F.direct.create_problem_description_deus_direct")
@patch("mu_F.direct.network_simulator")
@patch("mu_F.direct.visualiser")
def test_apply_direct_method(mock_vis, mock_net_sim, mock_create_prob, mock_construct):
    # Setup
    G = nx.DiGraph()
    G.add_node(0)
    
    mock_cfg = MagicMock()
    mock_cfg.case_study.design_space_dimensions = ['x1', 'x2']
    mock_cfg.reconstruction.post_process = False
    mock_cfg.reconstruction.plot_reconstruction = 'nominal_map'
    
    # Mock Solver
    mock_solver = MagicMock()
    # Mock feasible set return as tuple (query, prob) if that's expected by visualization logic
    mock_solver.get_solution.return_value = ((np.zeros((100,2)),), (np.zeros((100,1)),)) 
    mock_construct.return_value = mock_solver
    
    # Mock Network Simulator
    mock_sim_instance = MagicMock()
    mock_sim_instance.function_evaluations = {0: 100}
    mock_net_sim.return_value = mock_sim_instance

    # Run
    apply_direct_method(mock_cfg, G)

    # Assertions
    mock_construct.assert_called_once()
    mock_solver.solve.assert_called_once()
    assert G.nodes[0]['fn_evals'] == 100
    mock_vis.assert_called()