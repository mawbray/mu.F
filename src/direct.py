
import jax.numpy as jnp
import numpy as np
import logging 
import pandas as pd
from unit_evaluators.constructor import network_simulator
from samplers.constructor import construct_deus_problem_network
from constraints.constructor import constraint_evaluator
from samplers.utils import create_problem_description_deus_direct
from visualisation.visualiser import visualiser
from reconstruction.constructor import reconstruction as reconstruct
from reconstruction.objects import live_set, dataset


from deus import DEUS

def apply_direct_method(cfg, graph):

    model = network_simulator(cfg, graph, constraint_evaluator)
    problem_description = create_problem_description_deus_direct(cfg, graph)
    solver =  construct_deus_problem_network(DEUS, problem_description, model)
    solver.solve()
    feasible_set, infeasible_set = solver.get_solution()
    logging.info(f"Feasible set shape: {feasible_set[0].shape}, Infeasible set shape: {infeasible_set[0].shape}")
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

    graph.graph['feasible_set'] = feasible_set
    visualiser(cfg, graph, data=df, string='design_space', path=f'design_space_direct').run()

    if cfg.reconstruction.post_process:

        def sampler(
            ):
            
            if isinstance(feasible_set, tuple):
                fs = feasible_set[0]
            else:
                fs = feasible_set
            rng = np.random.default_rng()
            rng.shuffle(fs, axis=0)
            n_l = cfg.samplers.ns.final_sample_live
            n_samples = cfg.samplers.ns.n_replacements
            rng = np.random.default_rng()
            bounds = [np.zeros(1), np.ones(1)*n_l]
            unrounded_indices = rng.uniform(bounds[0], bounds[1], (n_samples, 1))
            rnd_ind = np.round(unrounded_indices).astype(int)
            rounded_indices = np.minimum(rnd_ind, n_l-1)
            feasible_samples = fs[rounded_indices]
            return feasible_samples
        
        graph =  load_to_graph(feasible_set, infeasible_set, graph, str_='post_process_lower_')
        post_process = graph.graph['post_process'](cfg, graph, model, 0)
        assert hasattr(post_process, 'run')
        # TODO: update this to allow flexibility for whether a sampling scheme or local SIP scheme is used
        post_process.load_training_methods(graph.graph["post_process_training_methods"])
        post_process.load_solver_methods(graph.graph["post_process_solver_methods"])
        post_process.graph.graph["solve_post_processing_problem"] = True
        post_process.sampler = lambda : sampler()
        post_process.load_fresh_live_set(live_set=live_set(cfg, cfg.samplers.notion_of_feasibility))
        graph = post_process.run()
        

    return feasible_set, infeasible_set


def load_to_graph(feasible, infeasible, graph, str_):
    assert isinstance(feasible, tuple) 
    assert isinstance(infeasible, tuple)
    # unpack feasible and infeasible sets
    feasible_query, feasible_prob = feasible
    infeasible_query, infeasible_prob = infeasible
    # get samples
    live_set = np.vstack(feasible_query)
    infeasible_set = np.vstack(infeasible_query) 
    # corresponding labels
    live_set_labels = np.ones(live_set.shape[0]).reshape(-1,1) 
    infeasible_set_labels = -np.ones(infeasible_set.shape[0]).reshape(-1,1)
    # create a dataset object
    all_data = np.vstack([live_set, infeasible_set])    
    all_labels = np.vstack([live_set_labels, infeasible_set_labels])
    logging.info(str_ + f"Live set size: {live_set.shape}, Infeasible set size: {infeasible_set.shape}")

    graph.graph[str_+ 'classifier_training'] = dataset(all_data, all_labels)
    return graph


