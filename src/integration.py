from abc import ABC
from functools import partial
from copy import deepcopy
from hydra.utils import get_original_cwd

import networkx as nx
import jax.numpy as jnp
import numpy as np
import logging
import jax.profiler as profiler
from jax import clear_backends, clear_caches

from constraints.constructor import constraint_evaluator
from unit_evaluators.constructor import subproblem_unit_wrapper
from constraints.solvers.surrogate.surrogate import surrogate
from samplers.constructor import construct_deus_problem
from samplers.appproximators import calculate_box_outer_approximation
from samplers.utils import create_problem_description_deus
from deus import DEUS
from utils import dataset_object as dataset_holder
from utils import dataset as dataset
from utils import data_processing as data_processor
from utils import apply_feasibility
from utils import save_graph
import time
import ray
import gc
from jax.lib import xla_bridge



class apply_decomposition:
    def __init__(self, cfg, graph, precedence_order, mode:str="forward", iterate=0, max_devices=1, total_iterates=1):
        self.cfg = cfg
        self.graph = graph
        self.precedence_order = precedence_order
        self.mode = mode
        self.iterate = iterate
        self.max_devices = max_devices
        self.total_iterates = total_iterates

    def run(self):

        cfg, graph = self.cfg, self.graph
        precedence_order, mode, iterate, max_devices, total_iterates = self.precedence_order, self.mode, self.iterate, self.max_devices, self.total_iterates


        if mode == "backward" or mode == "forward-backward":
            nodes = reversed(precedence_order.copy())
        elif mode == "forward" or mode == "backward-forward":
            nodes = precedence_order.copy()
        else:
            raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward' or 'forward-backward'.")
        

        # Iterate over the nodes and apply nested sampling
        for node in nodes:
            #if cfg.solvers.evaluation_mode.forward == 'ray':  ray.init(runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/']})
            logging.info(f'------- Characterising node {node} according to precedence order -------')
            # define model for deus
            model = subproblem_model(node, cfg, graph, mode=mode, max_devices=max_devices)
            # create problem sheet according to cfg 
            problem_sheet = create_problem_description_deus(cfg, model, graph, node, mode) 
            # solve extended DS using NS
            solver =  construct_deus_problem(DEUS, problem_sheet, model)
            solver.solve()
            if cfg.method == 'decomposition_constraint_tuner': graph.nodes[node]['log_evidence'] = solver.get_log_evidence()
            feasible, infeasible = solver.get_solution()
            feasible_set, feasible_set_prob = feasible[0], feasible[1]
            if feasible_set.size == 0:
                logging.warning(f"No feasible set found for node {node}. Terminating simulation.")
                graph.graph['terminate'] = True
                return graph
            # update the graph with the number of function evaluations
            graph.nodes[node]["fn_evals"] += model.function_evaluations
            # estimate box for bounds for DS downstream
            process_data_forward(cfg, graph, node, model, feasible_set, mode)
            # train constraints for DS downstream using data now stored in the graph
            if (mode in ['forward'] and graph.out_degree(node) != 0) or (mode in ['backward'] and graph.in_degree(node) == 0): 
                if cfg.surrogate.forward_evaluation_surrogate: surrogate_training_forward(cfg, graph, node)

            # train reward function
            if cfg.case_study.eval_rewards and graph.out_degree(node) != 0:
                graph = get_q_training_data(graph, node, model, cfg)

            # classifier construction for current unit
            if (cfg.surrogate.classifier and mode != 'backward-forward'): classifier_construction(cfg, graph, node, iterate) # NOTE this is a study specific condition
            if (cfg.surrogate.probability_map and mode != 'backward-forward'): probability_map_construction(cfg, graph, node, iterate) #  NOTE this is a study specific condition
            if (cfg.case_study.eval_rewards and mode != 'backward-forward'): q_function_construction(cfg, graph, node, iterate)

            del model, problem_sheet, solver, infeasible, feasible_set_prob, feasible_set, feasible
            
            
            save_graph(graph.copy(), mode + '_iterate_' + str(iterate)+ '_node_' + str(node))
            graph = del_data(graph, node)
            gc.collect()
            profiler.save_device_memory_profile(f"memory{node}.prof")
            clear_caches()
            clear_backends()
            profiler.save_device_memory_profile(f"memory{node}_post_backend_clear.prof")
        """ except: 
            if cfg.method == 'decomposition_constraint_tuner': graph.nodes[node]['log_evidence'] = {'mean':-10, 'std':0}
            gc.collect()
            profiler.save_device_memory_profile(f"memory{node}.prof")
            clear_caches()
            clear_backends()
            profiler.save_device_memory_profile(f"memory{node}_post_backend_clear.prof")"""

        return graph

    
def del_data(graph, node):

    del graph.nodes[node]["classifier_training"]
    graph.nodes[node]["classifier_training"] = None

    for successor in graph.successors(node):
        if 'surrogate_training' in graph.edges[node, successor]:
            del graph.edges[node, successor]["surrogate_training"]
            graph.edges[node, successor]["surrogate_training"] = None
    return graph



def surrogate_training_forward(cfg, graph, node, iterate:int=0):
    """
    Train the surrogate model for the forward pass.
    - only train the node if it has successors
    - train the node with the input data from the current node
    - store the trained model in the graph

    """

    # Check if the node has successors
    if graph.out_degree()[node] == 0:
        return
    
    for successor in graph.successors(node):
        # train the model
        forward_evaluator_surrogate = surrogate(graph, node, cfg, ('regression', cfg.surrogate.regressor_selection, 'forward_evaluation_surrogate'), iterate)
        forward_evaluator_surrogate.fit(node=successor)
        if cfg.solvers.standardised:
            query_model = forward_evaluator_surrogate.get_model('standardised_model')
        else:
            query_model = forward_evaluator_surrogate.get_model('unstandardised_model')
        
        # store the trained model in the graph
        graph.edges[node, successor]["forward_surrogate"] = query_model
        graph.nodes[node]['x_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_input')
        graph.edges[node,successor]['y_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_output')
        graph.edges[node,successor]["forward_surrogate_serialised"] = forward_evaluator_surrogate.get_serailised_model_data()
    
    return query_model



def probability_map_construction(cfg, graph, node, iterate):
    """
    Construct the probability map for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'probability_map' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """
    # train the model
    ls_surrogate = surrogate(graph, node, cfg, ('regression', 'ANN', 'probability_map_surrogate'), iterate)
    ls_surrogate.fit(node=None)
    if cfg.surrogate.probability_map_args.standardised:
        query_model = ls_surrogate.get_model('standardised_model')
    else:
        query_model = ls_surrogate.get_model('unstandardised_model')
    
    # store the trained model in the graph
    graph.nodes[node]["probability_map"] = query_model
    graph.nodes[node]['probability_map_x_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.edges[node]['probability_map_y_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_output')

    return

def get_classifier_data(graph, node, model, cfg):
    x_d, y_d = model.constraint_data.d[cfg.surrogate.index_on:], model.constraint_data.y[cfg.surrogate.index_on:]
    x_classifier, y_classifier, feasible_indices = apply_feasibility(x_d , y_d , cfg, node, cfg.formulation).get_feasible(return_indices = True)
    graph.nodes[node]["classifier_training"] = dataset(X=x_classifier, y=y_classifier) 

    return graph, feasible_indices

def get_q_training_data(graph, node, model, cfg):
    x_d_list = model.q_values.d[cfg.surrogate.index_on:]
    y_q_list = model.q_values.y[cfg.surrogate.index_on:]
    y_d_list = model.constraint_data.y[cfg.surrogate.index_on:]

    # Apply feasibility per batch and collect feasible samples
    x_feasible_list = []
    y_feasible_list = []

    for x_batch, y_q_batch, y_d_batch in zip(x_d_list, y_q_list, y_d_list):
        _, _, feasible_indices = apply_feasibility([x_batch], [y_d_batch], cfg, node, cfg.formulation).get_feasible(return_indices=True)

        # feasible_indices[0] because apply_feasibility returns list
        x_feasible_list.append(x_batch[feasible_indices[0]])
        y_feasible_list.append(y_q_batch[feasible_indices[0]])

    # Concatenate all feasible samples
    x_feasible = jnp.vstack(x_feasible_list) if x_feasible_list else jnp.empty((0, x_d_list[0].shape[1]))
    y_feasible = jnp.vstack(y_feasible_list) if y_feasible_list else jnp.empty((0, y_q_list[0].shape[1]))

    graph.nodes[node]["q_func_training"] = dataset(X=x_feasible, y=y_feasible)
    return graph

def get_probability_map_data(graph, node, model, cfg):
    x_d, y_d = model.probability_map_data.d[cfg.surrogate.index_on:], model.probability_map_data.y[cfg.surrogate.index_on:]
    x_classifier, y_classifier, feasible_indices = apply_feasibility(x_d , y_d, cfg, node, cfg.formulation).probabilistic_feasibility(return_indices = True)
    graph.nodes[node]["probability_map_training"] = dataset(X=x_classifier, y=y_classifier) 

    return graph

def process_data_forward(cfg, graph, node, model, live_set, mode, notion_of_feasibility='positive'):
    """
    Process the data in the forward direction.

    Parameters:
    cfg (object): The configuration object with a 'classification_threshold' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.
    model (object): The model object with 'input_output_data' and 'constraint_data' attributes.
    live_set (jnp.array): The live set data.

    Returns:
    None
    """
    # Select a subset of the data based on the classifier
    if (mode != 'backward-forward'):
        if cfg.surrogate.classifier:
            graph, feasible_indices = get_classifier_data(graph, node, model, cfg)
        elif cfg.surrogate.probability_map:
            graph = get_probability_map_data(graph, node, model, cfg)
        else:
            pass
        
        # Here insert logic for saving the surrogate data to the node

        # Apply the selected function to the y data and store forward evaluations on the graph for surrogate training
        for successor in graph.successors(node):

            # --- apply edge function to output data --- #
            io_fn = graph.edges[node, successor]["edge_fn"]

            # Could defined a reward function that takes it from the graph functions like the edge functions.
            reward_fn = graph.edges[node, successor]["reward_fn"]

            # --- select the approximation method
            if cfg.surrogate.forward_evaluation_surrogate:
                # Extract the input-output and classifier data from the model
                x_io, y_io, selected_y_io = data_processor(model.input_output_data, index_on = cfg.surrogate.index_on).transform_data_to_matrix(io_fn, feasible_indices) 
                if cfg.formulation == 'deterministic':
                    n_args = graph.nodes[node]['n_design_args'] + graph.nodes[node]['n_input_args']
                    x_io = x_io[:,:n_args]
            
            
                # --- apply the function to the selected output data --- #
                # ensure the output data is rank 2
                if y_io.ndim > 2: y_io= y_io.squeeze()
                if selected_y_io.ndim < 2: selected_y_io= selected_y_io.reshape(-1,1)
                # --- select the approximation method
                if cfg.samplers.ku_approximation == 'box': 
                    feasible_outer_approx = calculate_box_outer_approximation
                elif cfg.samplers.ku_approximation == 'ellipsoid':
                    raise NotImplementedError("Ellipsoid approximation not implemented yet.")

                # --- find box bounds on inputs
                graph.edges[node, successor][
                    "input_data_bounds"
                ] = feasible_outer_approx(selected_y_io, cfg, ndim=2)

                # store the forward evaluations on the graph for surrogate training
                forward_evals = dataset(X=x_io, y=y_io)
                graph.edges[node, successor]["surrogate_training"] = forward_evals 

                del x_io, y_io, selected_y_io, forward_evals
    # add live set to the node    
    graph.nodes[node]["live_set_inner"] = live_set    
    # Store the classifier data and the live set data to the node
    update_aux_bounds(live_set, graph, node, cfg)
    update_node_bounds_iplus1(graph, node, cfg)

    del live_set

    return

def update_aux_bounds(live_set, graph, node, cfg):
    # AUX SET IS THE INTERSECTION SO CAN DO THIS AFTER EACH NODE
    if cfg.case_study.n_aux_args[f'node_{node}'] != 0:
        aux = live_set[:,-cfg.case_study.n_aux_args[f'node_{node}']:]
        aux_bounds = calculate_box_outer_approximation(aux, cfg, ndim=2)
        graph.graph['aux_bounds'] = [[[aux_bounds[0][0,i], aux_bounds[1][0,i]]] for i in range(aux_bounds[0].shape[1])]
    else: pass

    return 



def transform_vmap_output(vmap_output):
    return jnp.vstack([vmap_output[:,i,:].squeeze() for i in range(vmap_output.shape[1])]) # transform rank3 tensor to rank 2 tensor ammenable for box approximation and surrogate training

def update_node_bounds_iplus1(graph, node, cfg):
    """
    Update the bounds of the node in the graph for iterate i+1 based on the liveset of the node at iterate i.

    Parameters:
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """

    # Get the bounds of the node i
    new_bounds = calculate_box_outer_approximation(graph.nodes[node]["live_set_inner"], cfg, ndim=2)
    graph.nodes[node]['extendedDS_bounds'] = new_bounds

    return

def q_function_construction(cfg, graph, node, iterate):
    # train the model
    q_surrogate = surrogate(graph, node, cfg, ('regression', cfg.surrogate.regressor_selection, 'q_function_surrogate'), iterate)
    q_surrogate.fit(node=None)
    if cfg.solvers.standardised:
        query_model = q_surrogate.get_model('standardised_model')
    else:
        query_model = q_surrogate.get_model('unstandardised_model')

    # store the trained model in the graph
    graph.nodes[node]["q_function"] = query_model
    graph.nodes[node]['q_function_x_scalar'] = q_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['q_function_serialised'] = q_surrogate.get_serialised_model_data()

    del q_surrogate
    
    return

def classifier_construction(cfg, graph, node, iterate):
    """
    Construct the classifier for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'classifier' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    classifier: The trained classifier. (-1 belongs to feasible region, 1 does not belong to feasible region)
    """
    # train the model
    ls_surrogate = surrogate(graph, node, cfg, ('classification', cfg.surrogate.classifier_selection, 'live_set_surrogate'), iterate)
    ls_surrogate.fit(node=None)
    if cfg.solvers.standardised:
        query_model = ls_surrogate.get_model('standardised_model')
    else:
        query_model = ls_surrogate.get_model('unstandardised_model')
    
    # store the trained model in the graph
    graph.nodes[node]["classifier"] = query_model
    graph.nodes[node]['classifier_x_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['classifier_serialised'] = ls_surrogate.get_serailised_model_data()

    del ls_surrogate

    return 


class subproblem_model(ABC):
    def __init__(self, unit_index, cfg, G, mode, max_devices):     
        """
        Class to construct the subproblem model for the DEUS solver.
        """
          
        # function evaluations, graph and unit-index intiialisation
        self.function_evaluations = 0
        self.unit_index = unit_index
        self.cfg, self.G = cfg, G

        # subproblem construction
        self.process_constraints = constraint_evaluator(cfg, G, unit_index, pool=None, constraint_type='process')
        if mode == 'forward':
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.forward, constraint_type='forward')
            self.backward_constraints = None
            self.forward_decentralised = None
            self.root_node_constraint = None
        elif mode == 'backward':
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.backward, constraint_type='backward')
            self.forward_constraints = None
            self.forward_decentralised = None
            self.root_node_constraint = None
            if self.cfg.case_study.eval_rewards is True:
                self.q_func_evalutor = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.reward, constraint_type='reward')
        elif (mode in ['forward-backward','backward-forward']):
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.forward, constraint_type='forward')
            if self.cfg.method == 'decomposition_constraint_tuner': 
                self.backward_constraints = None
                self.forward_decentralised = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.forward, constraint_type='forward_decentralized')
                if (mode == 'backward-forward'):
                    self.root_node_constraint = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.forward, constraint_type='root_node_decentralized')
                else:
                    self.root_node_constraint = None
            else: 
                self.backward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.backward, constraint_type='backward')
                self.forward_decentralised = None
                self.root_node_constraint = None
        else:
            raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward', 'forward-backward' or 'backward-forward' .")
        
        # decentralised?

        # subproblem unit construction
        self.unit_forward_evaluator = subproblem_unit_wrapper(cfg, G, unit_index)
        if (cfg.method == 'decomposition_constraint_tuner') and (mode != 'backward'):
            self.unit_forward_evaluator.get_constraints = lambda d, p: None
        else:
            pass

        # dataset initialisation 
        self.input_output_data = None 
        self.constraint_data = None 
        self.probability_map_data = None
        self.q_values = None
        self.mode = mode
        self.max_devices = max_devices
        self.hold_outputs = []


    def determine_batches(self, data, batch_size):
        """ Method to determine the number of batches"""
        n_batches = data.shape[0] // batch_size
        if data.shape[0] % batch_size != 0:
            n_batches += 1
        return n_batches
    
    def evaluate_subproblem_batch(self, data, batch_size, p, hold = False):
        """ Method to evaluate the subproblem in batches"""
        n_batches = self.determine_batches(data, batch_size)
        constraints = []
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size,:]
            results, output = self.subproblem_constraint_evals(batch, p)
            constraints.append(results)
            if hold:
                self.hold_outputs.append(output)
            del batch
            if (data.shape[0]>50) and (n_batches % (i+1) == 0): logging.info(f'Batch {i} of {n_batches} evaluated')

        return np.concatenate(constraints, axis=0)
    
    def evaluate_q_target_batch(self, data, batch_size, p):
        """Method to evaluatate q-learning in batches"""
        n_batches = self.determine_batches(data, batch_size)
        q_functions = []
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size,:]
            q_functions.append(self.q_target_evaluation(batch, p, self.hold_outputs[i]))
            del batch
            if (data.shape[0]>50) and (n_batches % (i+1) == 0): logging.info(f'Reward batch {i} of {n_batches} evaluated')

    def subproblem_constraint_evals(self, d, p):
        # unit forward pass
        outputs = self.unit_forward_evaluator.get_constraints(d, p) # outputs (rank 3 tensor if we have parametric uncertainty in the unit, n_d \times n_theta \times n_g)

        # get design/inputs/aux parameters split
        unit_design, unit_inputs, aux_args = self.unit_forward_evaluator.get_auxilliary_input_decision_split(d) # decisions, inputs (both rank 2 tensors)

        # evaluate process constraints 
        if outputs is not None:
            process_constraint_evals = self.process_constraints.evaluate(unit_design, unit_inputs, aux_args, outputs) # process constraints (rank 3 tensor n_d \times n_theta \times n_g)
        else:
            process_constraint_evals = None

        # evaluate feasibility upstream
        if (self.forward_constraints is not None) and (self.G.in_degree(self.unit_index) > 0):
            start_time = time.time()
            forward_constraint_evals = self.forward_constraints.evaluate(unit_inputs, aux_args) # forward constraints (rank 3 tensor n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f'execution_time_forward_constraints: {execution_time}')
            del start_time, end_time, execution_time
            if forward_constraint_evals.ndim == 1:
                forward_constraint_evals = forward_constraint_evals.reshape(-1,1)
            if forward_constraint_evals.ndim == 2:
                forward_constraint_evals = np.expand_dims(forward_constraint_evals, axis=1)
            forward_constraint_evals = np.repeat(forward_constraint_evals, len(p), axis=1)
        else:
            forward_constraint_evals = None
        
        # evaluate feasibility downstream
        if (self.backward_constraints is not None) and (self.G.out_degree(self.unit_index) > 0):
            start_time = time.time()
            backward_constraint_evals = self.backward_constraints.evaluate(outputs, aux_args) # backward constraints (rank 3 tensor, n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            if backward_constraint_evals.ndim == 1:
                backward_constraint_evals = backward_constraint_evals.reshape(-1,1)
            if backward_constraint_evals.ndim == 2:
                backward_constraint_evals = np.expand_dims(backward_constraint_evals, axis=1)
            logging.info(f'execution_time_backward_constraints: {execution_time}')
        else:
            backward_constraint_evals = None

        # evaluate feasibility decentralised
        if self.forward_decentralised is not None and self.G.in_degree(self.unit_index) > 0:
            start_time = time.time()
            decentralised_constraint_evals = self.forward_decentralised.evaluate(unit_design, aux_args) 
            if decentralised_constraint_evals.ndim == 1:
                decentralised_constraint_evals = decentralised_constraint_evals.reshape(-1,1)
            if decentralised_constraint_evals.ndim == 2:
                decentralised_constraint_evals = np.expand_dims(decentralised_constraint_evals, axis=1)
            decentralised_constraint_evals= np.repeat(decentralised_constraint_evals, len(p), axis=1)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f'execution_time_decentralised_constraints: {execution_time}')
        else:
            decentralised_constraint_evals = None

        # evaluate root node feasibility 
        if self.root_node_constraint is not None and self.G.in_degree(self.unit_index) == 0:
            start_time = time.time()
            decentralised_root_constraint_evals = self.root_node_constraint.evaluate(unit_design, aux_args) 
            if decentralised_root_constraint_evals.ndim == 1:
                decentralised_root_constraint_evals = decentralised_root_constraint_evals.reshape(-1,1)
            if decentralised_root_constraint_evals.ndim == 2:
                decentralised_root_constraint_evals = np.expand_dims(decentralised_root_constraint_evals, axis=1)
            decentralised_root_constraint_evals= np.repeat(decentralised_root_constraint_evals, len(p), axis=1)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f'execution_time_decentralised_constraints: {execution_time}')
        else:
            decentralised_root_constraint_evals = None


        # update input output data for forward surrogate model
        # TODO check all this works, turn off data collection for forward on decentrlaised and check backoffs
        if (self.cfg.surrogate.forward_evaluation_surrogate and self.mode != 'backward-forward'):
            self.input_output_data = update_data(self.input_output_data, d, p, outputs)  # updating dataset for surrogate model of forward unit evaluation

        # concatenate constraint evaluations
        concat_obj = [process_constraint_evals, forward_constraint_evals, backward_constraint_evals, decentralised_constraint_evals, decentralised_root_constraint_evals]
        cons_g = jnp.concatenate([c for c in concat_obj if c is not None], axis=-1)  # return raw constraint values (n_d \times n_theta \times n_g)

        # storing classifier data and updating function evaluations
        if (self.cfg.surrogate.classifier and self.mode != 'backward-forward'):
            self.constraint_data = update_data(self.constraint_data, d, p, cons_g)  # updating dataset for surrogate model of forward unit evaluation
        if (self.cfg.surrogate.probability_map and self.mode != 'backward-forward'):
            self.probability_map_data = update_data(self.probability_map_data, d, p, self.SAA(cons_g))  # updating dataset for surrogate model of forward unit evaluation



        del process_constraint_evals, forward_constraint_evals, backward_constraint_evals, concat_obj, outputs, decentralised_constraint_evals

        return cons_g, outputs

    def q_target_evaluation(self, d, p, output):
        rewards = self.unit_forward_evaluator.get_rewards(d,p)
        _, _, aux_args = self.unit_forward_evaluator.get_auxilliary_input_decision_split(d)
        start_time = time.time()
        q_function_evals = self.q_func_evalutor.evaluate(output, aux_args)
        q_learning_target = rewards + self.cfg.case_study.discount_factor * q_function_evals
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f'execution_time_q_function: {execution_time}')
        self.q_values = update_data(self.q_values, d, p, q_learning_target)

    def s(self, d, p):
        if (self.forward_constraints is not None) and (self.G.in_degree(self.unit_index) > 0) and (self.cfg.solvers.evaluation_mode.forward == 'ray'):
            ray.init(runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/', '../.git/']}, num_cpus=10)  # , ,
        # evaluate feasibility and then update classifier data and number of function evaluations
        g = self.evaluate_subproblem_batch(d, self.max_devices, p)
        
        # We need to init ray now to perform the optimisation over the reward fu
        if (self.config.case_study.eval_rewards and self.G.out_degree(self.unit_index) > 0 and self.cfg.solvers.evaluation_mode.reward == 'ray'):
            ray.init(runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/', '../.git/']}, num_cpus=10)  # , ,

        self.evaluate_q_target_batch(d, p)

        # shape parameters for returning constraint evaluations to DEUS
        n_theta, n_g = g.shape[-2], g.shape[-1]
        # adding function evaluations
        self.function_evaluations += g.shape[0]*g.shape[1]
        # return information for DEUS
        if (self.forward_constraints is not None) and (self.G.in_degree(self.unit_index) > 0)  and (self.cfg.solvers.evaluation_mode.forward == 'ray'):
            ray.shutdown()
        return [g[i,:,:].reshape(n_theta,n_g) for i in range(g.shape[0])]
        
    def get_constraints(self, d, p):
        return self.s(d, p)
    
    def SAA(self, g):
        
        if self.cfg.samplers.notion_of_feasibility == 'positive':
            g_ = jnp.min(g, axis=-1).reshape(g.shape[0],g.shape[1])
            indicator = jnp.where(g_>=0, 1, 0)
        else:
            g_ = jnp.max(g, axis=-1).reshape(g.shape[0],g.shape[1])
            indicator = jnp.where(g_<=0, 1, 0)

        n_s = g.shape[1]
        prob_feasible = jnp.sum(indicator, axis=1)/n_s
            
        return prob_feasible.reshape(g.shape[0],1)
    

    
def update_data(data, *args):
    """ Method to update the data holder with new data"""
    if data is None:
        data = dataset_holder(*args)
    else:
        data.add(*args)

    return data
