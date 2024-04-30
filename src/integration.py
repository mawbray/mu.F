from abc import ABC
from functools import partial

import gpjax as gpx
import networkx as nx
import jax.numpy as jnp
import numpy as np
import logging

from constraints.constructor import constraint_evaluator
from unit_evaluators.constructor import subproblem_unit_wrapper, network_simulator
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from visualisation.visualiser import visualiser
from surrogate.surrogate import surrogate
from solvers.constructor import solver_construction
from samplers.constructor import construct_deus_problem
from samplers.appproximators import calculate_box_outer_approximation
from samplers.space_filling import sobol_sample_design_space_nd
from samplers.utils import create_problem_description_deus
from deus import DEUS
from utils import dataset_object as dataset_holder
from utils import data_processing as data_processor
from utils import apply_feasibility

def apply_nested_sampling(cfg, graph, mode:str="forward", max_devices=1):
    # TODO:
    # redefine the constraints to return direct constraint evaluations and not indicator values.
    # implement the methods to reconstruct
    
    # Create a list of nodes in the graph according to the precedence order
    nodes = nx.topological_sort(graph)


    if mode == "backward" or mode == "forward-backward":
        nodes = reversed(list(nodes))
    elif mode == "forward":
        nodes = list(nodes)
    else:
        raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward' or 'forward-backward'.")
        


    # Iterate over the nodes and apply nested sampling
    for node in nodes:
        logging.info(f'------- Characterising node {node} according to precedence order: {nodes} -------')
        # define model for deus
        model = ModelA(node, cfg, graph, mode=mode, notion_of_feasibility=cfg.notion_of_feasibility, evaluation_mode=cfg.evaluation_mode, max_devices=max_devices)
        # create problem sheet according to cfg
        problem_sheet = create_activity_form(cfg, model, graph, node) 
        # solve extended DS using NS
        live_set, deadpoints_all, model = run_deus_nested_sampling(
            problem_sheet, model, cfg, graph, node
        )
        # estimate box for bounds for DS downstream
        process_data_forward(cfg, graph, node, model, live_set)
        # train constraints for DS downstream using data now stored in the graph
        if mode == 'forward': surrogate_training_forward(cfg, graph, node)
        # classifier construction for current unit
        classifier_construction(cfg, graph, node)


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
        if cfg.surrogate.forward_evaluation_surrogate.standardised:
            query_model = forward_evaluator_surrogate.get_model('standardised_model')
        else:
            query_model = forward_evaluator_surrogate.get_model('unstandardised_model')
        
        # store the trained model in the graph
        graph.edges[node, successor]["forward_surrogate"] = query_model
        graph.nodes[node]['x_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_input')
        graph.edges[node,successor]['y_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_output')

    return query_model



def process_data_forward(cfg, graph, node, model, live_set, notion_of_feasibility='positive'):
    """
    Process the data in the forward direction.

    Parameters:
    cfg (object): The configuration object with a 'classification_threshold' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.
    model (object): The model object with 'input_output_data' and 'classifier_data' attributes.
    live_set (jnp.array): The live set data.

    Returns:
    None
    """

    # Extract the input-output and classifier data from the model
    x_io, y_io = data_processor(model.input_output_data).transform_data_to_matrix() 
    x_classifier, y_classifier = data_processor(model.classifier_data).transform_data_to_matrix() # rename data to constraints
    x_prob, y_prob = data_processor(model.probability_map_data).transform_data_to_matrix() # rename data to constraints

    # Select a subset of the data based on the classifier    TODO - implement selection of data depending on whether we want a classifier, a regressor or a probability map, switch on or switch of probability map storage.
    selected_x, selected_y = apply_feasibility(x_classifier, y_classifier).get_feasible()
    selected_px, selected_py = apply_feasibility(x_prob, y_prob).get_feasible()

    # Apply the selected function to the y data and store forward evaluations on the graph
    for successor in graph.successors(node):
        # --- apply edge function to output data --- #
        io_fn = graph.edges[node, successor]["edge_fn"]
        # --- apply the function to the selected output data --- #
        y_updated_io = io_fn(selected_y) # should be rank 3 tensor with (nd, n_theta, n_g)
        y_updated = transform_vmap_output(y_updated_io)
        # ensure the output data is rank 2
        if y_updated.ndim > 2: y_updated= y_updated.squeeze()
        if y_updated.ndim < 2: y_in_node= y_in_node.reshape(-1,1)
        # --- select the approximation method
        if cfg.approximation == 'box': 
             feasible_outer_approx = calculate_box_outer_approximation
        elif cfg.approximation == 'ellipsoid':
            raise NotImplementedError("Ellipsoid approximation not implemented yet.")

        # --- find box bounds on inputs
        graph.edges[node, successor][
            "input_data_bounds"
        ] = feasible_outer_approx(y_updated, cfg)

        # store the forward evaluations on the graph for surrogate training
        y_in_node = io_fn(y_io)
    

        if y_in_node.ndim > 2: y_in_node= y_in_node.squeeze()
        if y_in_node.ndim < 2: y_in_node= y_in_node.reshape(-1,1)

        forward_evals = dataset_holder(X=x_io, y=y_in_node)
        graph.edges[node, successor]["surrogate_training"] = forward_evals # store the forward evaluations on the graph for surrogate training

    # Store the classifier data and the live set data to the node
    graph.nodes[node]["classifier_training"] = model.classifier_data
    graph.nodes[node]["live_set_inner"] = live_set

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
    new_bounds = calculate_box_outer_approximation(graph.nodes[node]["live_set_inner"], cfg)
    graph.nodes[node]['extendedDS_bounds'] = new_bounds

    return


def classifier_construction(cfg, graph, node):
    """
    Construct the classifier for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'classifier' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    classifier: The trained classifier. (-1 belongs to feasible region, 1 does not belong to feasible region)
    """

    

    return construct_coupling_constraint(graph, node, cfg)



class subproblem_model(ABC):
    def __init__(self, unit_index, cfg, G, mode, evaluation_mode, max_devices):     
        """
         TODO 
            1 - validate data storage and retrieval
            2 - validate constraint evaluations under uncertainty
        """
          
        # function evaluations, graph and unit-index intiialisation
        self.function_evaluations = 0
        self.unit_index = unit_index
        self.cfg, self.G = cfg, G

        # subproblem construction
        self.process_constraints = constraint_evaluator(cfg, G, unit_index, cfg.constraints.pool, 'process')
        if mode == 'forward':
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, cfg.constraints.pool, 'forward')
            self.backward_constraints = None
        elif mode == 'backward':
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, cfg.constraints.pool, 'backward')
            self.forward_constraints = None
        elif mode == 'forward-backward':
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, cfg.constraints.pool, 'forward')
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, cfg.constraints.pool, 'backward')

        # subproblem unit construction
        self.unit_forward_evaluator = subproblem_unit_wrapper(cfg, G, unit_index, mode)

        # dataset initialisation 
        self.input_output_data = None 
        self.constraint_data = None 
        self.probability_map_data = None
        self.mode = mode
        self.evaluation_mode = evaluation_mode
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

        return np.vstack(constraints)
        
    def subproblem_constraint_evals(self, d, p):
        # unit forward pass
        outputs = self.unit_forward_evaluator.get_constraints(d, p) # outputs (rank 3 tensor if we have parametric uncertainty in the unit, n_d \times n_theta \times n_g)

        # get inputs/design parameters split
        unit_design, unit_inputs = self.unit_forward_evaluator.get_input_decision_split(d) # decisions, inputs (both rank 2 tensors)

        # evaluate process constraints 
        process_constraint_evals = self.process_constraints.evaluate(unit_design, unit_inputs, outputs) # process constraints (rank 3 tensor n_d \times n_theta \times n_g)

        # evaluate feasibility upstream
        if self.forward_constraints is not None:
            forward_constraint_evals = self.forward_constraints.evaluate(unit_inputs) # forward constraints (rank 3 tensor n_d \times n_theta \times n_g)
        else:
            forward_constraint_evals = None
        
        # evaluate feasibility downstream
        if self.backward_constraints is not None:
            backward_constraint_evals = self.backward_constraints.evaluate(outputs) # backward constraints (rank 3 tensor, n_d \times n_theta \times n_g)
        else:
            backward_constraint_evals = None

        # update input output data for forward surrogate model
        self.input_output_data = update_data(self.input_output_data, d, p, outputs, d_axis=-1, p_axis=-1, y_axis=-1)  # updating dataset for surrogate model of forward unit evaluation

        return jnp.concatenate([process_constraint_evals, forward_constraint_evals, backward_constraint_evals], axis=-1)  # return raw constraint values (n_d \times n_theta \times n_g)

    def s(self, d, p):
        # evaluate feasibility and then update classifier data and number of function evaluations
        g = self.evaluate_subproblem_batch(d, self.max_devices, p.reshape(1,1,-1))
        # shape parameters for returning constraint evaluations to DEUS
        n_theta, n_g = g.shape[-2], g.shape[-1]
        # storing classifier data and updating function evaluations
        self.classifier_data = update_data(self.classifier_data, d, g)  # updating dataset for surrogate model of forward unit evaluation
        self.probability_map_data = update_data(self.probability_map_data, d, self.SAA(g))  # updating dataset for surrogate model of forward unit evaluation
        # adding function evaluations
        self.function_evaluations += g.shape[0]*g.shape[1]
        # return information for DEUS
        return [g[i,:,:].reshape(n_theta,n_g) for i in range(g.shape[0])]
        
    def get_constraints(self, d, p):
        return self.s(d, p)
    
    def SAA(self, constraints):
        if self.cfg.notion_of_feasibility == 'positive':
            return jnp.mean(jnp.cond(jnp.max(constraints, axis=-1) >= 0 , 1, 0), axis=1).expand_dims(axis=1).expand_dims(axis=1)
        else:
            return jnp.mean(jnp.cond(jnp.max(constraints, axis=-1) <= 0 , 1, 0), axis=1).expand_dims(axis=1).expand_dims(axis=1)
        
    
def update_data(data, *args):
    """ Method to update the data holder with new data"""
    if data is None:
        data = dataset_holder(*args)
    else:
        data.add(*args)

    return data
