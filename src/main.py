from integration import *
from cs_assembly import * 
from utils import *

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
    G = case_study_constructor(cfg)
    # Save the graph to a file
    save_graph(G.copy(), "initial")
    # iterate over the modes defined in the config file
    mode = cfg.mode
    for i, m in enumerate(mode):
        if (i == 0) and not (m == 'forward'):
            G = initialise(cfg, G, max_devices) # initialise the extended knowledge space with forward simulations
        G = apply_nested_sampling(cfg, G, mode=m, max_devices=max_devices)
        save_graph(G.copy(), m + '_iterate_' + str(i))
        if cfg.reconstruct[i]:
            ModelA, _, _ = initialise_cs2(cfg)
            model = ModelA()
            joint_live_set = joint_reconstruction(G, cfg, model, max_devices)
            for node in G.nodes():
                G.nodes[node]["fn_evals"] += model.function_evaluations
            # DATAFRAME FOR VISUALISATION
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(cfg.design_space_dimensions)})
            design_space_visualisation(df)
            df.to_excel(f'inside_samples_{mode}_iterate_{i}.xlsx')
            save_graph(G.copy(), m + '-reconstructed'+ '_iterate_' + str(i))


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