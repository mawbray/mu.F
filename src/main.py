from integration import apply_decomposition
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from visualisation.visualiser import visualiser
from unit_evaluators.constructor import network_simulator
from constraints.constructor import constraint_evaluator
from samplers.space_filling import sobol_sampler
from samplers.appproximators import calculate_box_outer_approximation as approximator 

from cs_assembly import * 
from utils import *

import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
import networkx as nx

"""
TODO :
- integration of all code module
- test and debugging
- documentation
"""


@hydra.main(config_path="config", config_name="tablet_press")
def main(cfg: DictConfig) -> None:
    # Set the maximum number of devices
    max_devices = len(jax.devices('cpu'))

    # Construct the case study graph
    G = case_study_constructor(cfg)   # TODO integration of case study construction G is a networkx graph - need to update case study contructor

    # Save the graph to a file
    save_graph(G.copy(), "initial")

    # iterate over the modes defined in the config file
    mode = cfg.mode

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
        if cfg.reconstruct[i]:
            network_model = network_model(cfg, G, constraint_evaluator)
            joint_live_set = reconstruction(G, cfg, network_model, max_devices) # TODO update uncertainty evaluations
            
            # update the graph with the function evaluations
            for node in G.nodes():
                G.nodes[node]["fn_evals"] += network_model.function_evaluations
            
            # visualisation of reconstruction
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.design_space_dimensions)})
            visualiser(cfg, G, df, 'reconstruction', path=f'reconstruction_{m}_iterate_{i}').visualise()
            df.to_excel(f'inside_samples_{mode}_iterate_{i}.xlsx')
            save_graph(G.copy(), m + '-reconstructed'+ '_iterate_' + str(i))

    # The following loop logs the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    print("Platform: ", platform)
    print(jax.devices('cpu'))

    # Enable 64 bit floating point precision
    jax.config.update("jax_enable_x64", True)
    main()
    print("Done")