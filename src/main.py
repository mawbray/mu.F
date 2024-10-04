import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from jax import jit

from integration import apply_decomposition
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from visualisation.visualiser import visualiser
from unit_evaluators.constructor import network_simulator
from constraints.constructor import constraint_evaluator
from samplers.space_filling import sobol_sampler
from samplers.appproximators import calculate_box_outer_approximation as approximator 
from direct import apply_direct_method
from functools import partial
from samplers.algorithms.bo.alg import bayesian_optimization
from cs_assembly import case_study_constructor
from utils import *
from samplers.space_filling import measure_live_set_volume  

import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
import networkx as nx

"""
TODO :
- visualisation of probability maps
- test and debugging
- documentation
"""

@partial(jit, static_argnums=(1,3))
def constraint_backoff(dynamics, cfg, xi, constraint_fn):
    return constraint_fn(dynamics, cfg) - xi  # g() >= xi 

def update_constraint_tuning_parameters(G, xi, G_init):
    """
    Update the constraint tuning parameters in the graph.
    """
    k = 0
    for node in G.nodes():
        if not (G.out_degree(node) == 0):
            for i, constraint in enumerate(G_init.nodes[node]['constraints']):
                xi_input  = jnp.array(xi[k]).squeeze()
                G.nodes[node]['constraints'][i] = partial(constraint_backoff, xi=xi_input, constraint_fn=constraint)
                k += 1
    return G


def run_a_single_evaluation(xi, cfg, G, G_init):
    # Set the maximum number of devices
    max_devices = len(jax.devices('cpu'))   

    # update the constraint parmeters.
    G = update_constraint_tuning_parameters(G, xi, G_init)

    # getting precedence order
    precedence_order = list(nx.topological_sort(G))
    m = 'backward-forward'
    G = apply_decomposition(cfg, G, precedence_order, mode=m, max_devices=max_devices)
    G.graph['iterate'] += 1

    # visualisation of decomposition
    visualiser(cfg, G, string='decomposition', path=f'decomposition_{m}_iterate_{G.graph["iterate"]}').visualise()
    save_graph(G.copy(), m + '_iterate_' + str(G.graph['iterate']))

    # The following loop logs the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")
    

    return measure_live_set_volume(G, cfg.case_study.design_space_dimensions)


@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:

    max_devices = len(jax.devices('cpu'))
    # Construct the case study graph
    G = case_study_constructor(cfg)   # TODO integration of case study construction G is a networkx graph - need to update case study contructor

    if cfg.method == 'direct':
        # apply direct method
        feasible, infeasible = apply_direct_method(cfg, G)
        (joint_live_set, joint_live_set_prob) = feasible
         # visualisation of reconstruction
        if cfg.reconstruction.plot_reconstruction == 'nominal_map':
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
        elif cfg.reconstruction.plot_reconstruction == 'probability_map':
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
            df['probability'] = joint_live_set_prob
        visualiser(cfg, G, data=df, string='design_space', path=f'design_space_direct').visualise()
        save_graph(G.copy(), 'direct_complete')

    elif cfg.method == 'decomposition':
        G = initialisation(cfg, G, network_simulator, constraint_evaluator, sobol_sampler(), approximator).run() # TODO update uncertainty evaluations
        # visualisation of initialisation
        visualiser(cfg, G, string='initialisation', path=f'initialisation_backward_iterate_0').visualise()
        save_graph(G.copy(), "initial")
        # getting precedence order
        precedence_order = list(nx.topological_sort(G))

        # decomposition
        m = 'backward'
        G = apply_decomposition(cfg, G, precedence_order, mode=m, max_devices=max_devices)

        # visualisation of decomposition
        visualiser(cfg, G, string='decomposition', path=f'decomposition_backward_iterate_0').visualise()
        save_graph(G.copy(), m + '_iterate_' + str(0))

        G_init = G.copy()
        G.graph['iterate'] = 0

        fn = partial(run_a_single_evaluation, cfg=cfg, G=G, G_init=G_init)

        lower_bound = [0]
        upper_bound = [0.04]
        num_initial_points = 4
        num_iterations = 5

        xi_opt, best_index = bayesian_optimization(fn, lower_bound, upper_bound, num_initial_points, num_iterations)


        logging.info("------- Finished -------")
        logging.info("Best candidate: {}".format(xi_opt))
        logging.info("Best index: {}".format(best_index))
        logging.info("------------------------")
    
    

    return 


if __name__ == "__main__":
    
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    platform = jax.lib.xla_bridge.get_backend().platform.casefold()

    # Enable 64 bit floating point precision
    jax.config.update("jax_enable_x64", True)
    # run the program
    main()
    print("Done")