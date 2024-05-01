
from omegaconf import DictConfig
import networkx as nx
import hydra
import numpy as np


def design_list_constructor(bounds_for_design):
    """ Method to construct a list of bounds for the design space"""

    bounds = {}
    for i, bound in enumerate(bounds_for_design):
        bounds[f'd{i+1}'] = {f'd{i+1}': [bound[0], bound[1]]}
        
    return bounds

def extended_design_list_constructor(bounds_for_input, bounds_for_design):
    """ Method to construct a list of bounds for the design space
    bounds for design is a nested list
    bounds for input is a dictonary
    
    """
    if len(bounds_for_input) > 0:
        bounds = {}
        n_index = len(bounds_for_design)
        for j in range(len(bounds_for_input)): # iterate over input streams
            n_input_shape = max(bounds_for_input[j][0].shape)
            for i in range(n_input_shape): # iterate over input variables for each stream
                bounds[f'd{n_index+1}'] = {f'd{n_index+1}': [bounds_for_input[j][0][0,i].squeeze(), bounds_for_input[j][1][0,i].squeeze()]}
                n_index +=1 
        return bounds_for_design | bounds
    else:
        return bounds_for_design

def get_unit_bounds(G: nx.DiGraph, unit_index: int):
    # constructing holder for input and design parameter bounds
    if G.nodes[unit_index]['extendedDS_bounds'] == 'None':
        design_var = design_list_constructor(G.nodes[unit_index]['KS_bounds'])
        if G.in_degree()[unit_index] > 0: 
            bounds_for_input = [G.edges[predec,unit_index]["input_data_bounds"] for predec in G.predecessors(unit_index)]
            bounds =  extended_design_list_constructor(bounds_for_input, design_var) # this should just operate on data in the graph.
        else:
            bounds = design_var
    else: 
        bounds = { f'd{index+1}': {f'd{index+1}': [ G.nodes[unit_index]['extendedDS_bounds'][0][0,index],  G.nodes[unit_index]['extendedDS_bounds'][1][0,index]]} for index in range(len(G.nodes[unit_index]['extendedDS_bounds'][0].squeeze()))}
    
    return bounds


def create_problem_description_deus(cfg: DictConfig, the_model: object, G:nx.DiGraph, unit_index:float, forward_mode:bool = False):
    

    # This is a problem description generation method specific to DEUS
    bounds = get_unit_bounds(G, unit_index)

    print(f"Unit index: {unit_index}")
    print(f"Bounds: {bounds}")
    print(f'EXTENDED DS DIM.: {len(bounds)}')
        
    the_activity_form = {
        "activity_type": cfg.samplers.deus.activity_type,
        "activity_settings": {
            "case_name": f"{cfg.case_study.case_study}_{unit_index}_fwd_{forward_mode}",
            "case_path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "resume": False, #' in general following the implementation here, we cannot load the problem via pickle
            "save_period": 1
        },

        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": G.nodes[unit_index]['parameters_best_estimate'],
            "parameters_samples": G.nodes[unit_index]['parameters_samples'],
            "target_reliability": cfg.samplers.target_reliability,
            "design_variables": [bound for bound in bounds.values()]
        },

        "solver": {
            "name": "dsc-ns",
            "settings": {
            "score_evaluation": {
                "method": "serial",
                "score_type": "sigmoid",  # "indicator",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False
            },
            # "score_evaluation": {
            #     "method": "mppool",
            #     "score_type": "sigmoid",  # "indicator",
            #     "pool_size": -1,
            #     "store_constraints": False
            # },
            "efp_evaluation": {
                "method": "serial",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False,
            },
            #"efp_evaluation": {
            #    "method": "mppool",
            #    "pool_size": -1,
            #    "store_constraints": False
            #},
            "phases_setup": {
                "initial": {
                    "nlive": cfg.samplers.ns.n_live,
                    "nproposals": cfg.samplers.ns.n_replacements
                },
                "deterministic": {
                    "skip": True
                },
                "nmvp_search": {
                    "skip": True
                },
                "probabilistic": {
                    "skip": False,
                    "nlive_change": {
                        "mode": "user_given",
                        "schedule": [
                            (.00, cfg.samplers.ns.n_live, cfg.samplers.ns.n_replacements),
                        ]
                    }
                }
            }
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": cfg.samplers.ns.n_live,  # This is overridden by points_schedule
                     "nproposals": cfg.samplers.ns.n_replacements,  # This is overriden by points_schedule
                     "prng_seed": 1989,
                     "f0": cfg.samplers.ns.f0,
                     "alpha": cfg.samplers.ns.alpha,
                     "stop_criteria": [
                         {"max_iterations": 100000}
                     ],
                     "debug_level": 0,
                     "monitor_performance": False
                 },
                "algorithms": {
                    "replacement": {
                        "sampling": {
                            "algorithm": "suob-ellipsoid"
                            }
                        }
                    }
                }
            }
        }
    }

    return the_activity_form