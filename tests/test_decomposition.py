import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from mu_F.decomposition import decomposition

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.init.sampler = 'sobol'
    cfg.samplers.ku_approximation = 'box'
    cfg.reconstruction.reconstruct = [False, False] # For 2 iterations
    cfg.method = 'decomposition'
    return cfg

@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    G.add_nodes_from([1,2])
    G.add_edge(1, 2)
    G.graph['terminate'] = False
    return G

@patch("mu_F.decomposition.apply_decomposition")
@patch("mu_F.decomposition.visualiser")
@patch("mu_F.decomposition.sobol_sampler")
def test_decomposition_initialization(mock_sobol, mock_vis, mock_apply, mock_cfg, simple_graph):
    decomp = decomposition(mock_cfg, simple_graph, precedence_order=list(simple_graph.nodes), mode=['forward'])
    assert decomp.total_iterations == 1
    assert decomp.precedence_order == list(simple_graph.nodes)
    mock_sobol.assert_called_once()

@patch("mu_F.decomposition.apply_decomposition")
@patch("mu_F.decomposition.visualiser")
@patch("mu_F.decomposition.sobol_sampler")
@patch("mu_F.decomposition.save_graph")
def test_decomposition_run_forward(mock_save, mock_sobol, mock_vis, mock_apply, mock_cfg, simple_graph):
    # Setup mock for apply_decomposition().run() to return the graph
    mock_runner = MagicMock()
    mock_runner.run.return_value = simple_graph
    mock_apply.return_value = mock_runner

    decomp = decomposition(mock_cfg, simple_graph, precedence_order=list(simple_graph.nodes), mode=['forward'])
    result_G = decomp.run()

    assert result_G == simple_graph
    # Verify operations were defined and run
    assert mock_apply.call_count == 1 # once per mode item
    mock_runner.run.assert_called()
