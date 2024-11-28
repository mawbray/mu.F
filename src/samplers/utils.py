
from omegaconf import DictConfig
import networkx as nx
import hydra
import logging
import jax.numpy as jnp
from jax.random import PRNGKey, choice

def design_list_constructor(bounds_for_design):
    """ Method to construct a list of bounds for the design space"""

    bounds = {}
    for i, bound in enumerate(bounds_for_design):
        if bound[0] != 'None': bounds[f'd{i+1}'] = {f'd{i+1}': [bound[0], bound[1]]}
        
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
    
def add_global_aux(bounds, G):
    if len(bounds) > 0:
        n_index = len(bounds)
    else: 
        n_index = 0

    aux_bounds = G.graph['aux_bounds']
    for j in range(len(aux_bounds)): # iterate over auxiliary args
        for n in aux_bounds[j]: # iterate over variables for each auxiliary arg
            if n[0] != 'None': 
                bounds[f'd{n_index+1}'] = {f'd{n_index+1}': [n[0], n[1]]}
                n_index +=1
    return bounds


def get_unit_bounds(G: nx.DiGraph, unit_index: int):
    # constructing holder for input and design parameter bounds
    if G.nodes[unit_index]['extendedDS_bounds'] == 'None':
        design_var = design_list_constructor(G.nodes[unit_index]['KS_bounds'])
        if G.in_degree()[unit_index] > 0: 
            bounds_for_input = [G.edges[predec,unit_index]['aux_filter'](G.edges[predec,unit_index]["input_data_bounds"]) for predec in G.predecessors(unit_index)]
            bounds =  extended_design_list_constructor(bounds_for_input, design_var) # this should just operate on data in the graph.
        else:
            bounds = design_var
        bounds = add_global_aux(bounds, G)
    else: 
        bounds = { f'd{index+1}': {f'd{index+1}': [ G.nodes[unit_index]['extendedDS_bounds'][0][0,index],  G.nodes[unit_index]['extendedDS_bounds'][1][0,index]]} for index in range(len(G.nodes[unit_index]['extendedDS_bounds'][0].squeeze()))}
    
    return bounds


def get_network_bounds(G: nx.DiGraph):

    bounds = []
    for node in G.nodes():
        bounds = bounds + G.nodes[node]['KS_bounds']

    dict_bounds = design_list_constructor(bounds)
    bounds = add_global_aux(dict_bounds, G)

    return dict_bounds


def create_problem_description_deus(cfg: DictConfig, the_model: object, G:nx.DiGraph, unit_index:float, forward_mode:bool = False):
    

    # This is a problem description generation method specific to DEUS
    bounds = get_unit_bounds(G, unit_index)

    logging.info(f"Bounds: {bounds}")
    logging.info(f'EXTENDED DS DIM.: {len(bounds)}')

    if cfg.formulation == 'deterministic': 
        parameter_samples = [{'c': jnp.array(G.nodes[unit_index]['parameters_best_estimate']).reshape(-1,), 'w': 1.0}]
    elif cfg.formulation == 'probabilistic':
        parameter_samples = [{key: value for key,value in dict_.items()} for i,dict_ in enumerate(G.nodes[unit_index]['parameters_samples']) if i < cfg.max_uncertain_samples]
        sum_weights = jnp.sum(jnp.array([param['w'] for param in parameter_samples ])) # normalise weights to sum to 1
        for param in parameter_samples:
            param['w'] = param['w']/sum_weights
    else: 
        raise ValueError('Formulation not recognised')

        
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
            "parameters_best_estimate": [el for el in G.nodes[unit_index]['parameters_best_estimate']],
            "parameters_samples": parameter_samples,
            "target_reliability": cfg.samplers.unit_wise_target_reliability[unit_index],
            "design_variables": [bound for bound in bounds.values()]
        },

        "solver": {
            "name": "dsc-ns",
            "settings": {
            "log_evidence_estimation": {"enabled": cfg.samplers.ns.log_evidence_estimation}, # must ve a boolean
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


def get_network_uncertain_params(cfg):
    param_dict = cfg.case_study.parameters_samples

    # getting uncertain parameters
    max_parameter_samples = min(cfg.max_uncertain_samples, min([len(param) for param in param_dict]))
    list_of_params, list_of_weights = {i: {} for i in range(len(param_dict)) }, {i: {} for i in range(len(param_dict)) }
    
    for i, param in enumerate(param_dict):
        for k in range(max_parameter_samples):
            list_of_params[i][k] = param[k]['c']
            list_of_weights[i][k] = param[k]['w']

    concat_params = [jnp.hstack([jnp.array(list_of_params[i][k]).reshape(1,-1) for i in range(len(param_dict))]) for k in range(max_parameter_samples)]
    prod_weights = [jnp.prod(jnp.hstack([jnp.array(list_of_weights[i][k]).reshape(1,1) for i in range(len(param_dict))])) for k in range(max_parameter_samples)]
    sum_prod_weights = jnp.sum(jnp.array(prod_weights))
    prod_weights = [prod_weights[i]/sum_prod_weights for i in range(max_parameter_samples)]
    
    list_ = []

    for i in range(max_parameter_samples):
        list_.append({'c': concat_params[i].squeeze(), 'w': prod_weights[i]})

    # getting nominal parameters
    nom_params = jnp.hstack([jnp.array(p).reshape(1,-1) for p in cfg.case_study.parameters_best_estimate]).squeeze()

    nom_p = [{'c': nom_params.squeeze(), 'w': 1.0}]

    return list_, nom_params, nom_p


def create_problem_description_deus_direct(cfg: DictConfig, G:nx.DiGraph):
    

    # This is a problem description generation method specific to DEUS
    bounds = get_network_bounds(G)
    uncertain_params, nom_params, n_p_samples = get_network_uncertain_params(cfg)


    if cfg.formulation == 'deterministic': 
        parameter_samples = n_p_samples
    elif cfg.formulation == 'probabilistic':
        parameter_samples = uncertain_params
    else:
        raise ValueError('Formulation not recognised')

    logging.info(f"Bounds: {bounds}")
    logging.info(f'DS DIM.: {len(bounds)}')

        
    the_activity_form = {
        "activity_type": cfg.samplers.deus.activity_type,
        "activity_settings": {
            "case_name": f"{cfg.case_study.case_study}_direct_mode",
            "case_path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "resume": False, #' in general following the implementation here, we cannot load the problem via pickle
            "save_period": 1
        },

        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": [nom_params[i] for i in range(len(nom_params))],
            "parameters_samples": parameter_samples,
            "target_reliability": cfg.samplers.target_reliability,
            "design_variables": [bound for bound in bounds.values()]
        },

        "solver": {
            "name": "dsc-ns",
            "settings": {
            "log_evidence_estimation": {"enabled": cfg.samplers.ns.log_evidence_estimation}, 
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
                    "nlive": cfg.samplers.ns.final_sample_live,
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
                            (.00, cfg.samplers.ns.final_sample_live, cfg.samplers.ns.n_replacements),
                        ]
                    }
                }
            }
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": cfg.samplers.ns.final_sample_live,  # This is overridden by points_schedule
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