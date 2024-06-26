import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from integration import apply_decomposition
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from visualisation.visualiser import visualiser
from unit_evaluators.constructor import network_simulator
from constraints.constructor import constraint_evaluator
from samplers.space_filling import sobol_sampler
from samplers.appproximators import calculate_box_outer_approximation as approximator 
from direct import apply_direct_method

from cs_assembly import case_study_constructor
from utils import *

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


@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    # Set the maximum number of devices
    max_devices = len(jax.devices('cpu'))

    # Construct the case study graph
    G = case_study_constructor(cfg)   # TODO integration of case study construction G is a networkx graph - need to update case study contructor

    # Save the graph to a file
    save_graph(G.copy(), "initial")


    if cfg.method == 'decomposition':

        # iterate over the modes defined in the config file
        mode = cfg.case_study.mode

        # getting precedence order
        precedence_order = list(nx.topological_sort(G))

        for i, m in enumerate(mode):
            # initialisation
            if (i == 0) and not (m == 'forward'):
                G = initialisation(cfg, G, network_simulator, constraint_evaluator, sobol_sampler(), approximator).run() # TODO update uncertainty evaluations
                # visualisation of initialisation
                visualiser(cfg, G, string='initialisation', path=f'initialisation_{m}_iterate_{i}').visualise()
            
            # decomposition
            G = apply_decomposition(cfg, G, precedence_order, mode=m, max_devices=max_devices)

            # visualisation of decomposition
            visualiser(cfg, G, string='decomposition', path=f'decomposition_{m}_iterate_{i}').visualise()
            save_graph(G.copy(), m + '_iterate_' + str(i))

            # reconstruction
            if cfg.reconstruction.reconstruct[i]: # TODO fix reconstruction vmap
                network_model = network_simulator(cfg, G, constraint_evaluator)
                joint_live_set, joint_live_set_prob = reconstruction(cfg, G, network_model).run() # TODO update uncertainty evaluations
                
                # update the graph with the function evaluations
                for node in G.nodes():
                    G.nodes[node]["fn_evals"] += network_model.function_evaluations
                
                # visualisation of reconstruction
                if cfg.reconstruction.plot_reconstruction == 'nominal_map':
                    df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
                elif cfg.reconstruction.plot_reconstruction == 'probability_map':
                    df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
                    df['probability'] = joint_live_set_prob
                visualiser(cfg, G, df, 'reconstruction', path=f'reconstruction_{m}_iterate_{i}').visualise()
                df.to_excel(f'inside_samples_{mode}_iterate_{i}.xlsx')
                save_graph(G.copy(), m + '-reconstructed'+ '_iterate_' + str(i))

            # TODO generalise this to all graphs based on in-degree and out-degree
            # update precedence order, note this should only be done for acyclic graphs
            if m == 'backward':
                for node in G.nodes():
                    if G.in_degree(node) == 0:
                        precedence_order.remove(node)
            elif m == 'forward':
                for node in G.nodes():
                    if G.out_degree(node) == 0:
                        precedence_order.remove(node)
                    
    elif cfg.method == 'direct':

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





    # The following loop logs the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    platform = jax.lib.xla_bridge.get_backend().platform.casefold()

    # Enable 64 bit floating point precision
    jax.config.update("jax_enable_x64", True)
    # run the program
    main()
    print("Done")