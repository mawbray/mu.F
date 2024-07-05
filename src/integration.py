from abc import ABC
from functools import partial
from copy import deepcopy
from hydra.utils import get_original_cwd

import networkx as nx
import jax.numpy as jnp
import numpy as np
import logging

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
import time
import ray


def apply_decomposition(cfg, graph, precedence_order, mode:str="forward", iterate=0, max_devices=1):

    if mode == "backward" or mode == "forward-backward":
        nodes = reversed(precedence_order.copy())
    elif mode == "forward" or mode == "backward-forward":
        nodes = precedence_order.copy()
    else:
        raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward' or 'forward-backward'.")
        

    # Iterate over the nodes and apply nested sampling
    for node in nodes:
        if cfg.solvers.evaluation_mode.forward == 'ray':  ray.init(runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/']})
        logging.info(f'------- Characterising node {node} according to precedence order -------')
        # define model for deus
        model = subproblem_model(node, cfg, graph, mode=mode, max_devices=max_devices)
        # create problem sheet according to cfg 
        problem_sheet = create_problem_description_deus(cfg, model, graph, node, mode) 
        # solve extended DS using NS
        solver =  construct_deus_problem(DEUS, problem_sheet, model)
        solver.solve()
        if cfg.solvers.evaluation_mode.forward == 'ray': ray.shutdown()
        feasible, infeasible = solver.get_solution()
        feasible_set, feasible_set_prob = feasible[0], feasible[1]
        # update the graph with the number of function evaluations
        graph.nodes[node]["fn_evals"] += model.function_evaluations
        # estimate box for bounds for DS downstream
        process_data_forward(cfg, graph, node, model, feasible_set)
        # train constraints for DS downstream using data now stored in the graph
        if (mode in ['forward', 'forward-backward', 'backward-forward']) or (mode in ['backward'] and graph.in_degree(node) == 0): surrogate_training_forward(cfg, graph, node)
        # classifier construction for current unit
        if cfg.surrogate.classifier: classifier_construction(cfg, graph, node, iterate)
        if cfg.surrogate.probability_map: probability_map_construction(cfg, graph, node, iterate)

        del model, problem_sheet, solver
        
        graph = del_data(graph, node)




    return graph

def del_data(graph, node):


    del graph.nodes[node]["classifier_training"]
    graph.nodes[node]["classifier_training"] = None

    for successor in graph.successors(node):
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
        forward_evaluator_surrogate = surrogate(graph, node, cfg, ('regression', 'ANN', 'forward_evaluation_surrogate'), iterate)
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


def process_data_forward(cfg, graph, node, model, live_set, notion_of_feasibility='positive'):
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
    x_d, y_d = model.constraint_data.d[cfg.surrogate.index_on:], model.constraint_data.y[cfg.surrogate.index_on:]
    x_classifier, y_classifier, feasible_indices = apply_feasibility(x_d , y_d , cfg, node, cfg.formulation).get_feasible(return_indices = True)
    graph.nodes[node]["classifier_training"] = dataset(X=x_classifier, y=y_classifier) 

    if cfg.surrogate.probability_map:
        # in the case of probabistic constraints, this will provide feasible data with P level set by the user.
        graph.nodes[node]["probability_map_training"] = dataset(X=x_classifier, y=y_classifier) # saves the whole probability map training data
    
    # add live set to the node    
    graph.nodes[node]["live_set_inner"] = live_set

    # Apply the selected function to the y data and store forward evaluations on the graph for surrogate training
    for successor in graph.successors(node):

        # --- apply edge function to output data --- #
        io_fn = graph.edges[node, successor]["edge_fn"]

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

    # Store the classifier data and the live set data to the node
    update_node_bounds_iplus1(graph, node, cfg)

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
        elif mode == 'backward':
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.backward, constraint_type='backward')
            self.forward_constraints = None
        elif (mode in ['forward-backward','backward-forward']):
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.forward, constraint_type='forward')
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, pool=cfg.solvers.evaluation_mode.backward, constraint_type='backward')
        else:
            raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward', 'forward-backward' or 'backward-forward' .")
        # subproblem unit construction
        self.unit_forward_evaluator = subproblem_unit_wrapper(cfg, G, unit_index)

        # dataset initialisation 
        self.input_output_data = None 
        self.constraint_data = None 
        self.probability_map_data = None
        self.mode = mode
        self.max_devices = max_devices


    def determine_batches(self, data, batch_size):
        """ Method to determine the number of batches"""
        n_batches = data.shape[0] // batch_size
        if data.shape[0] % batch_size != 0:
            n_batches += 1
        return n_batches
    
    def evaluate_subproblem_batch(self, data, batch_size, p):
        """ Method to evaluate the subproblem in batches"""
        n_batches = self.determine_batches(data, batch_size)
        constraints = []
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size,:]
            constraints.append(self.subproblem_constraint_evals(batch, p))
            del batch
            if (data.shape[0]>50) and (n_batches % (i+1) == 0): logging.info(f'Batch {i} of {n_batches} evaluated')

        return np.concatenate(constraints, axis=0)
        
    def subproblem_constraint_evals(self, d, p):
        # unit forward pass
        outputs = self.unit_forward_evaluator.get_constraints(d, p) # outputs (rank 3 tensor if we have parametric uncertainty in the unit, n_d \times n_theta \times n_g)

        # get inputs/design parameters split
        unit_design, unit_inputs = self.unit_forward_evaluator.get_input_decision_split(d) # decisions, inputs (both rank 2 tensors)

        # evaluate process constraints 
        process_constraint_evals = self.process_constraints.evaluate(unit_design, unit_inputs, outputs) # process constraints (rank 3 tensor n_d \times n_theta \times n_g)

        # evaluate feasibility upstream
        if (self.forward_constraints is not None) and (self.G.in_degree(self.unit_index) > 0):
            start_time = time.time()
            forward_constraint_evals = self.forward_constraints.evaluate(unit_inputs) # forward constraints (rank 3 tensor n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f'execution_time_forward_constraints: {execution_time}')
            del start_time, end_time, execution_time
            if forward_constraint_evals.ndim == 1:
                forward_constraint_evals = forward_constraint_evals.reshape(-1,1)
            if forward_constraint_evals.ndim == 2:
                forward_constraint_evals = np.expand_dims(forward_constraint_evals, axis=1)
            forward_constraint_evals = np.repeat(forward_constraint_evals, outputs.shape[1], axis=1)
        else:
            forward_constraint_evals = None
        
        
        
        # evaluate feasibility downstream
        if (self.backward_constraints is not None) and (self.G.out_degree(self.unit_index) > 0):
            start_time = time.time()
            backward_constraint_evals = self.backward_constraints.evaluate(outputs) # backward constraints (rank 3 tensor, n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f'execution_time_backward_constraints: {execution_time}')
        else:
            backward_constraint_evals = None

        

        # update input output data for forward surrogate model

        if self.cfg.surrogate.forward_evaluation_surrogate:
            self.input_output_data = update_data(self.input_output_data, d, p, outputs)  # updating dataset for surrogate model of forward unit evaluation

        # concatenate constraint evaluations
        concat_obj = [process_constraint_evals, forward_constraint_evals, backward_constraint_evals]
        cons_g = jnp.concatenate([c for c in concat_obj if c is not None], axis=-1)  # return raw constraint values (n_d \times n_theta \times n_g)

        # storing classifier data and updating function evaluations
        if self.cfg.surrogate.classifier:
            self.constraint_data = update_data(self.constraint_data, d, p, cons_g)  # updating dataset for surrogate model of forward unit evaluation
        if self.cfg.surrogate.probability_map:
            self.probability_map_data = update_data(self.probability_map_data, d, p, self.SAA(cons_g))  # updating dataset for surrogate model of forward unit evaluation

        del process_constraint_evals, forward_constraint_evals, backward_constraint_evals, concat_obj, outputs

        return cons_g

    def s(self, d, p):
        # evaluate feasibility and then update classifier data and number of function evaluations
        g = self.evaluate_subproblem_batch(d, self.max_devices, p)
        # shape parameters for returning constraint evaluations to DEUS
        n_theta, n_g = g.shape[-2], g.shape[-1]
        # adding function evaluations
        self.function_evaluations += g.shape[0]*g.shape[1]
        # return information for DEUS
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
