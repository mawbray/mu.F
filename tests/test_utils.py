import pytest
import jax.numpy as jnp
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch
from mu_F.utils import save_graph, dataset_object, dataset, data_processing, apply_feasibility

@pytest.fixture
def sample_data():
    d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    p = jnp.array([[0.1], [0.2]])
    y = jnp.array([[[10.0]], [[20.0]]])
    return d, p, y

def test_dataset_object_initialization(sample_data):
    d, p, y = sample_data
    ds = dataset_object(d, p, y)
    
    # Based on code: d, p, y are stored in lists
    assert len(ds.d) == 1
    assert len(ds.p) == 1
    assert len(ds.y) == 1
    assert jnp.array_equal(ds.d[0].squeeze(), d)

def test_dataset_object_add(sample_data):
    d, p, y = sample_data
    ds = dataset_object(d, p, y)
    ds.add(d, p, y)
    
    assert len(ds.d) == 2
    assert len(ds.p) == 2
    assert len(ds.y) == 2

def test_data_processing_transformation(sample_data):
    d, p, y = sample_data
    ds = dataset_object(d, p, y)
    
    # Mock edge function (identity for simplicity)
    edge_fn = lambda x: x
    
    dp = data_processing(ds)
    X, Y, Z = dp.transform_data_to_matrix(edge_fn, feasible_indices=None)
    
    # Check output shapes
    # X should stack d and p. d=(2,2), p=(2,1). 
    assert X.shape[0] == 4 
    assert Y.shape[0] == 4

@patch("mu_F.utils.pickle")
@patch("builtins.open")
def test_save_graph(mock_open, mock_pickle):
    G = nx.DiGraph()
    G.add_node(1)
    G.nodes[1]["forward_evaluator"] = "should_be_removed"
    
    save_graph(G, "forward")
    
    # Verify cleanup happened
    assert G.nodes[1]["forward_evaluator"] is None
    # Verify pickle was called
    mock_pickle.dump.assert_called_once()

def test_apply_feasibility_deterministic():
    # Setup
    X = jnp.array([[1], [2]])
    Y = jnp.array([[[0.5]], [[-0.5]]]) # 1st infeasible (>0), 2nd feasible (<0) if notion is max<=0
    
    mock_cfg = MagicMock()
    mock_cfg.samplers.notion_of_feasibility = 'negative' # max <= 0
    mock_cfg.formulation = 'deterministic'
    
    af = apply_feasibility(X, Y, mock_cfg, node=1, feasibility='deterministic')
    _, _, cond = af.get_feasible(return_indices=True)
    
    # cond returns list of booleans/arrays
    assert not cond[0] # 0.5 <= 0 is False
    assert cond[1]     # -0.5 <= 0 is True