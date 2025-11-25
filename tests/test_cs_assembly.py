import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from mu_F.cs_assembly import case_study_constructor, case_study_allocation

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.case_study.case_study = 'test_study'
    cfg.case_study.adjacency_matrix = [[0, 1], [0, 0]]
    cfg.case_study.design_space_dimensions = ['x1']
    cfg.case_study.n_design_args = {0: 1, 1: 1}
    cfg.case_study.n_theta = {0: 0, 1: 0}
    cfg.case_study.KS_bounds.design_args = []
    cfg.case_study.parameters_best_estimate = {}
    cfg.case_study.parameters_samples = {}
    cfg.case_study.fn_evals = {0: 0, 1: 0}
    cfg.case_study.unit_op = {}
    cfg.case_study.extendedDS_bounds = {}
    cfg.case_study.n_input_args = {}
    cfg.case_study.n_aux_args = {}
    cfg.case_study.global_n_aux_args = 0
    cfg.case_study.KS_bounds.aux_args = []
    cfg.case_study.vmap_evaluations = False
    cfg.reconstruction.post_process = False
    cfg.surrogate.post_process_lower.model_class = 'none'
    cfg.method = 'standard'
    return cfg

@patch("mu_F.cs_assembly.CS_holder", {'test_study': {}})
@patch("mu_F.cs_assembly.CS_edge_holder", {'test_study': {}})
@patch("mu_F.cs_assembly.graph_constructor")
@patch("mu_F.cs_assembly.solver_construction")
@patch("mu_F.cs_assembly.unit_params_fn")
@patch("mu_F.cs_assembly.aux_filter")
@patch("mu_F.cs_assembly.unit_evaluation")
def test_case_study_constructor(mock_unit_eval, mock_aux, mock_params, mock_solver, mock_graph_ctor, mock_cfg):
    
    # Mock the graph constructor object and the graph it returns
    mock_G_obj = MagicMock()
    mock_real_graph = nx.DiGraph()
    mock_real_graph.add_nodes_from([0, 1])
    mock_real_graph.add_edge(0, 1)
    
    mock_G_obj.get_graph.return_value = mock_real_graph
    mock_G_obj.G = mock_real_graph # Often accessed directly
    mock_graph_ctor.return_value = mock_G_obj

    # Run
    G = case_study_constructor(mock_cfg)

    # Check allocation was called (via graph modifications)
    # n_design_args should be added to nodes
    mock_G_obj.add_arg_to_nodes.assert_any_call('n_design_args', mock_cfg.case_study.n_design_args)
    
    # Check graph return
    assert G == mock_real_graph