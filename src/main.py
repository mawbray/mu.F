import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
from visualisation.visualiser import visualiser

from direct import apply_direct_method
from decomposition import decomposition, decomposition_constraint_tuner
from cs_assembly import case_study_constructor
from graph.graph_assembly import build_graph_structure
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
    print(get_original_cwd())
    max_devices = len(jax.devices('cpu'))

    # Querying if the case study is a repeated single node
    if hasattr(cfg.case_study, 'serial_graph'):
        if cfg.case_study.serial_graph is True:
            cfg = build_graph_structure(cfg)


    # Construct the case study graph
    G = case_study_constructor(cfg)   # TODO integration of case study construction G is a networkx graph - need to update case study contructor

    # Save the graph to a file
    save_graph(G.copy(), "initial")

    # identify constraint sets
    if cfg.method == 'decomposition':
        # iterate over the modes defined in the config file
        mode = cfg.case_study.mode
        # getting precedence order
        precedence_order = list(nx.topological_sort(G))
        # run the decomposition
        G = decomposition(cfg, G, precedence_order, mode, max_devices).run()
        # finished decomposition                    
    elif cfg.method == 'direct':
        # run the decomposition
        feasible, infeasible = apply_direct_method(cfg, G)
        save_graph(G.copy(), 'direct_complete')
    elif cfg.method == 'decomposition_constraint_tuner':
        decomposition_constraint_tuner(cfg, G, max_devices)
    else:
        # raise an error
        raise ValueError("Method not recognised")
        
    # Log the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    import jax
    import sys
    from hydra.utils import get_original_cwd
    jax.config.update('jax_platform_name', 'cpu')
    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    
    # Enable 64 bit floating point precision
    #jax.config.update("jax_enable_x64", True)
    sys.path.append(os.path.join(os.getcwd(),'src'))
    # run the program
    main()
    print("Done")