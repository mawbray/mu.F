
import jax.numpy as jnp
import numpy as np
import pandas as pd
from unit_evaluators.constructor import network_simulator
from samplers.constructor import construct_deus_problem_network
from constraints.constructor import constraint_evaluator
from samplers.utils import create_problem_description_deus_direct
from visualisation.visualiser import visualiser
from reconstruction.constructor import reconstruction as reconstruct
from reconstruction.objects import live_set

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

    graph.graph['feasible_set'] = feasible_set
    visualiser(cfg, graph, data=df, string='design_space', path=f'design_space_direct').run()

    if cfg.reconstruction.post_process:

        def sampler(
            ):
            
            if isinstance(feasible_set, tuple):
                fs = feasible_set[0]
            else:
                fs = feasible_set
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
        
        post_process = graph.graph['post_process'](cfg, graph, model, 0)
        assert hasattr(post_process, 'run')
        post_process.load_training_methods(graph.graph["post_process_training_methods"])
        post_process.load_solver_methods(graph.graph["post_process_solver_methods"])
        post_process.graph.graph["solve_post_processing_problem"] = True
        post_process.sampler = lambda : sampler()
        post_process.load_fresh_live_set(live_set=live_set(cfg, cfg.samplers.notion_of_feasibility))
        graph = post_process.run()
        trainer = post_process.training_methods(graph, None, cfg, ('classification', cfg.surrogate.classifier_selection, 'live_set_surrogate'), 0,'post_process_classifier_training')
        trainer.fit()
        
        if cfg.solvers.standardised:
            query_model = trainer.get_model('standardised_model')
        else:
            query_model = trainer.get_model('unstandardised_model')
        
        # store the trained model in the graph
        graph.graph["final_post_process_classifier"] = query_model
        graph.graph['final_post_process_classifier_x_scalar'] = trainer.trainer.get_model_object('standardisation_metrics_input')
        graph.graph['final_post_process_classifier_serialised'] = trainer.get_serailised_model_data()
        

    

    return feasible_set, infeasible_set