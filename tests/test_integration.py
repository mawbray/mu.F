import pytest
import networkx as nx
import numpy as np
from unittest.mock import MagicMock, patch
from mu_F.integration import apply_decomposition, subproblem_model

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.solvers.evaluation_mode.forward = 'serial'
    cfg.surrogate.forward_evaluation_surrogate = False
    cfg.surrogate.classifier = False
    cfg.surrogate.probability_map = False
    cfg.method = 'standard'
    return cfg

@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    G.add_nodes_from([0,1])
    G.add_edge(0, 1)
    G.graph['terminate'] = False
    for node in G.nodes:
        G.nodes[node]["fn_evals"] = 0
    return G


@patch("mu_F.integration.subproblem_model")
def test_subproblem_model_batches(mock_wrapper, mock_cfg):
    # Test batch logic in subproblem_model
    G = nx.DiGraph()
    G.add_node(0)
    G.nodes[0]['unit_op'] = 'steady_state'
    G.nodes[0]['unit_params_fn'] = lambda x: x
    
    # Create instance directly (mocking internal evaluators via cfg/init)
    with patch("mu_F.integration.constraint_evaluator"):
        model = subproblem_model(0, mock_cfg, G, mode='forward', max_devices=1)
        
        # Test determine_batches
        data = np.zeros((100, 2))
        batches = model.determine_batches(data, batch_size=10)
        assert batches == 10
        
        batches_uneven = model.determine_batches(data, batch_size=30)
        assert batches_uneven == 4 # 3 full + 1 partial