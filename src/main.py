"""
Main running file. To run enter src and then run with the necessary flags for your case study...
"""

import os
import logging
import hydra
import pandas as pd
import networkx as nx
import argparse

from logging import config
from omegaconf import DictConfig

def _fix_devices(num_devices):
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return None

@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    # Fixing the devices
    _fix_devices(int(cfg.max_devices))
    
    # Import at this level so device setting happens first
    from direct import apply_direct_method
    from decomposition import decomposition, decomposition_constraint_tuner
    from cs_assembly import case_study_constructor
    from graph.graph_assembly import build_graph_structure
    from visualisation.visualiser import visualiser
    from utils import save_graph
   
    import jax
    
    # Set the maximum number of devices    
    cpu_devs = jax.devices("cpu")
    logging.info(f"Requested max_devices: {cfg.max_devices}")
    logging.info(f"JAX CPU device count: {len(cpu_devs)}")
    logging.info(f"All devices: {jax.devices()}")

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
        G = decomposition(cfg, G, precedence_order, mode, cfg.max_devices).run()
        # finished decomposition                    
    elif cfg.method == 'direct':
        # run the decomposition
        feasible, infeasible = apply_direct_method(cfg, G)
        save_graph(G.copy(), 'direct_complete')
    elif cfg.method == 'decomposition_constraint_tuner':
        decomposition_constraint_tuner(cfg, G, cfg.max_devices)
    else:
        # raise an error
        raise ValueError("Method not recognised")
        
    # Log the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    import sys
    import os

    sys.path.append(os.path.join(os.getcwd(),'src'))      
    
    main()
    print("Done")
