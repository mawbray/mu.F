import os
import multiprocessing

# 1. Force spawn immediately, BEFORE any other imports run
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# 2. Set JAX/Ray flags before JAX inits
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Prevent Ray/JAX deadlock logs if Ray is used
#os.environ["RAY_DEDUP_LOGS"] = "0"

import logging
import hydra
from omegaconf import DictConfig
import networkx as nx


"""
TODO :
- visualisation of probability maps
- extensive unit tests
- documentation
"""

@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    import ray
    from hydra.utils import get_original_cwd
    if not cfg.method == 'direct':
        ray.init(
            _node_ip_address="127.0.0.1",  
            include_dashboard=False, 
            runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/', '../.git/']},
            num_cpus=10)  # , ,
    
    # Set the maximum number of devices
    from mu_F.direct import apply_direct_method
    from mu_F.decomposition import decomposition, decomposition_constraint_tuner
    from mu_F.cs_assembly import case_study_constructor
    from mu_F.utils import save_graph
    import jax
    jax.config.update('jax_platform_name', 'cpu')

    print(get_original_cwd())
    max_devices = len(jax.devices('cpu'))

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
    
    if not cfg.method == 'direct':
        ray.shutdown()
        
    # Log the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    
    import sys  

    # Enable 64 bit floating point precision
    #jax.config.update("jax_enable_x64", True)
    sys.path.append(os.path.join(os.getcwd(),'src'))
    # run the program
    main()
    print("Done")