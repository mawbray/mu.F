
import jax.numpy as jnp
import pandas as pd
from unit_evaluators.constructor import network_simulator
from samplers.constructor import construct_deus_problem_network
from constraints.constructor import constraint_evaluator
from samplers.utils import create_problem_description_deus_direct
from visualisation.visualiser import visualiser

from deus import DEUS

def apply_direct_method(cfg, graph):

    model = network_simulator(cfg, graph, constraint_evaluator)
    problem_description = create_problem_description_deus_direct(cfg, graph)
    solver =  construct_deus_problem_network(DEUS, problem_description, model)
    solver.solve()
    feasible_set, infeasible_set = solver.get_solution()
    for node in graph.nodes:
        graph.nodes[node]['fn_evals'] = model.function_evaluations[node]
    
    if cfg.reconstruction.plot_reconstruction == 'nominal_map':
        if isinstance(feasible_set, tuple):
            df = pd.DataFrame({key: feasible_set[0][:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
        else:
            df = pd.DataFrame({key: feasible_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
    elif cfg.reconstruction.plot_reconstruction == 'probability_map':
        df = pd.DataFrame({key: feasible_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
        df['probability'] = feasible_set
    visualiser(cfg, graph, data=df, string='design_space', path=f'design_space_direct').run()
    

    print(feasible_set[0].max(axis=0))
    print(feasible_set[0].min(axis=0))
    return feasible_set, infeasible_set