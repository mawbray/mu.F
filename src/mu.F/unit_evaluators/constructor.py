
from abc import ABC
from copy import copy
from functools import partial
import jax.numpy as jnp 
from jax import vmap, jit
import numpy as np
import pandas as pd 
import logging

from unit_evaluators.integrators import unit_dynamics
from unit_evaluators.steady_state import unit_steady_state
from unit_evaluators.utils import arrhenius_kinetics_fn as arrhenius


class base_unit(ABC):
    def __init__(self, cfg, graph, node):
        self.cfg = cfg
        self.graph = graph
        self.node = node

    def get_decision_dependent_params(self, decisions):
        raise NotImplementedError

    def evaluate(self, decisions, x0):
        raise NotImplementedError

class unit_evaluation(base_unit):
    def __init__(self, cfg, graph, node):
        """
        Initializes the unit_evaluation object.

        Args:
            cfg (object): Configuration object containing various settings and parameters.
            graph (object): Graph object representing the structure of the system.
            node (str): The node in the graph for which this unit_evaluation object is being created.

        The method calls the base_unit's __init__ method and sets the unit_cfg object.
        """
        super().__init__(cfg, graph, node)
        self.unit_cfg = unit_cfg(cfg, graph, node)


    def get_decision_dependent_params(self, decisions, uncertain_params=None):
        """
        Returns the decision dependent parameters.

        Args:
            decisions (array): Array of decisions.

        Returns:
            array: Decision dependent parameters.
        """
        return self.unit_cfg.decision_dependent_params(decisions, uncertain_params)

    def evaluate(self, design_args, input_args, aux_args, uncertain_params=None):
        """
        Evaluates the unit.

        Args:
            design_args (array): Array of design arguments selected by sampler.
            input_args (array): Array of input arguments selected by sampler or by previous unit operation.

        Returns:
            array: The result of the evaluation.

        The method gets the decision dependent parameters using the design arguments, 
        concatenates them with the design arguments to form the system parameters, 
        and then evaluates the unit using these parameters.
        """

        dd_params = self.get_decision_dependent_params(design_args, uncertain_params)
        dd_params = expand_dims(dd_params, axis=-1)
        input_args = expand_dims(input_args, axis=1)
        design_args = expand_dims(design_args, axis=1)
        aux_args = expand_dims(aux_args, axis=1)            

        return self.unit_cfg.evaluator(design_args, input_args, aux_args, dd_params, uncertain_params)


def expand_dims(array, axis):
    if array.ndim < 3:
        array = jnp.expand_dims(array, axis=axis)
    return array


class subproblem_unit_wrapper(unit_evaluation):
    def __init__(self, cfg, graph, node):
        """
        Initializes the subproblem_unit_wrapper object.

        Args:
            cfg (object): Configuration object containing various settings and parameters.
            graph (object): Graph object representing the structure of the system.
            node (str): The node in the graph for which this subproblem_unit_wrapper object is being created.

        The method calls the unit_evaluation's __init__ method to initialize the object.
        """
        super().__init__(cfg, graph, node)

    def get_constraints(self, decisions, uncertain_params=None):
        """
        Returns the constraints for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            array: The constraints for the given decisions and uncertain parameters.

        The method splits the decisions into design arguments and input arguments based on the number of design arguments in the node, 
        and then evaluates the unit using these arguments and the uncertain parameters.
        """
        if uncertain_params is None:
            uncertain_params = jnp.empty((1,1))
        
        design_args, input_args, aux_args = self.get_auxilliary_input_decision_split(decisions)
        
        # if no inputs to the unit, use the root node inputs or add empty array
        if input_args.shape[1] == 0: 
            if not (self.cfg.model.root_node_inputs[self.node] == 'None'):
                input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]]*design_args.shape[0])
            else:
                input_args = jnp.empty((design_args.shape[0], 0))
        # if no inputs to the unit, use the root node inputs or add empty array
        if aux_args.shape[1] == 0: 
            if not (self.cfg.model.node_aux[self.node] == 'None'):
                aux_args = jnp.array([self.cfg.model.root_node_aux[self.node]]*design_args.shape[0])
            else:
                aux_args = jnp.empty((design_args.shape[0], 0))
            
        input_args = expand_input_args(input_args, uncertain_params)
        #aux_args = expand_input_args(aux_args, uncertain_params)
        
        return self.evaluate(design_args, input_args, aux_args, uncertain_params)
    
    def get_auxilliary_input_decision_split(self, decisions):
        """
        """
        n_d = self.graph.nodes[self.node]['n_design_args']
        n_u = self.graph.nodes[self.node]['n_input_args']
        design_args, input_args, auxiliary_args = decisions[:,:n_d], decisions[:,n_d:n_d+n_u], decisions[:,n_d+n_u:]
        return design_args, input_args, auxiliary_args

def expand_input_args(array, template):
    if array.ndim == 1:
        array = jnp.expand_dims(array, axis=1)
    if array.ndim < 3:
        array = jnp.expand_dims(array, axis=1)

    if array.shape[1] != template.shape[0]:
        array = jnp.concatenate([array for _ in range(template.shape[0])], axis=1) # repeat input_args for each uncertain param

    return array

class unit_cfg:
    def __init__(self, cfg, graph, node):
        """
        Initializes the unit_cfg object.

        Args:
            cfg (object): Configuration object containing various settings and parameters.
            graph (object): Graph object representing the structure of the system.
            node (str): The node in the graph for which this unit_cfg object is being created.

        Raises:
            NotImplementedError: If vmap of unit evaluation is enabled in cfg or if the unit operation or unit parameters function is not implemented.

        The method sets the unit evaluation function and the decision dependent evaluation function based on the node's attributes in the graph. 
        If the unit operation is 'dynamic', the unit evaluation function is set to unit_dynamics. 
        If the unit parameters function is 'Arrhenius', the decision dependent evaluation function is set to arrhenius. 
        If the unit parameters function is None, the decision dependent evaluation function is set to return an empty array of the same shape as the input.
        """

        self.cfg, self.graph, self.node = cfg, graph, node

        # if vmap is enabled in cfg, set the unit evaluation and decision dependent evaluation functions using vmap
        if cfg.case_study.vmap_evaluations:
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = vmap(vmap(jit(partial(unit_dynamics, cfg=cfg, node=node)), in_axes=(0, 0, 0, 0, None), out_axes=0), in_axes=(None, 1, None, 1, 0), out_axes=1) # inputs are design args, input args, deicsion_and_uncertainty_dependent_params, uncertain params
            elif graph.nodes[node]['unit_op'] == 'steady_state':
                self.evaluator = vmap(vmap(jit(partial(unit_steady_state, cfg=cfg, node=node)), in_axes=(0, 0, 0, 0, None), out_axes=0), in_axes=(None, 1, None, 1, 0), out_axes=1)   
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation 
            fn = graph.nodes[node]['unit_params_fn']
            self.decision_dependent_params = vmap(vmap(fn, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)

        # if vmap is not enabled in cfg, set the unit evaluation and decision dependent evaluation functions without using vmap
        else: 
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = lambda x, y, z: jit(partial(unit_dynamics, cfg=cfg, node=node))(x.squeeze(), y.squeeze(), z.squeeze())
            elif graph.nodes[node]['unit_op'] == 'steady_state':
                self.evaluator = lambda x, y, z: jit(partial(unit_steady_state, cfg=cfg, node=node))(x.squeeze(), y.squeeze(), z.squeeze())
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation fn
            fn = graph.nodes[node]['unit_params_fn']
            self.decision_dependent_params = fn

        return
    
class network_simulator(ABC):
    """
    Abstract base class for a network simulator.

    This class is responsible for simulating a network of interconnected nodes and edges. 
    Each node represents a unit operation in a process, and each edge represents the flow of material between units.
    """
    def __init__(self, cfg, graph, constraint_evaluator, type_cons='process'):
        self.cfg = cfg
        self.graph = graph.copy()
        self.type = type_cons
        self.constraint_evaluator = constraint_evaluator
        self.function_evaluations = {node: 0 for node in self.graph.nodes}

    def simulate(self, decisions, uncertain_params=None):
        """
        Simulates the network for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (list, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the nodes and the values are the constraints for each node.
            dict: A dictionary where the keys are the edges and the values are the input data for each edge.

        The method simulates the network by iterating over each node, evaluating the node, storing the output in the input data store of each successor edge, 
        and storing the constraints of the node in the constraint store of the node.
        """
        u_p = None
        n_d = 0
        aux_args = decisions[:, sum([self.graph.nodes[node]['n_design_args'] for node in self.graph.nodes]):]
        
        for node in self.graph.nodes:
            if not (uncertain_params == None) :
                u_p = uncertain_params[node]
            

            if self.graph.in_degree()[node] == 0:
                if not (self.cfg.model.root_node_inputs[node] == 'None'):
                    inputs = jnp.tile(jnp.expand_dims(jnp.array([self.cfg.model.root_node_inputs[node]]).reshape(1,-1), axis=1), (decisions.shape[0], u_p.shape[0], 1))
                else:
                    inputs = jnp.empty((decisions.shape[0], u_p.shape[0], 0))
            else:
                inputs = jnp.concatenate([jnp.copy(self.graph.edges[predecessor, node]['input_data_store'])[:,:,:] for predecessor in self.graph.predecessors(node)], axis=-1)

            unit_nd = self.graph.nodes[node]['n_design_args']
            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, u_p)
            
            for successor in self.graph.successors(node):
                edge_data = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))
                if edge_data.ndim==2: edge_data = jnp.expand_dims(edge_data, axis=-1)
                self.graph.edges[node, successor]['input_data_store'] = edge_data

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node, constraint_type=self.type)

            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, outputs)

            n_d += unit_nd

        # constraint evaluation, information for extended KS bounds
        return {node: self.graph.nodes[node]['constraint_store'] for node in self.graph.nodes}, {edge: self.graph.edges[edge[0],edge[1]]['input_data_store'] for edge in self.graph.edges}
    
    def get_constraints(self, decisions, uncertain_params=None):
        """
        Returns the constraints for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the nodes and the values are the constraints for each node.

        The method simulates the network and returns the constraints.
        """
        constraints, _ = self.simulate(decisions, uncertain_params)
        for node, g in constraints.copy().items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        return constraints
    
    def get_extended_ks_info(self, decisions, uncertain_params=None):
        """
        Returns the input data for each edge for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the edges and the values are the input data for each edge.

        The method simulates the network and returns the input data for each edge.
        """
        _, edge_data = self.simulate(decisions, uncertain_params)
        return edge_data

    def get_data(self, decisions, uncertain_params=None):
        """
        Returns the constraints and the input data for each edge for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the nodes and the values are the constraints for each node.
            dict: A dictionary where the keys are the edges and the values are the input data for each edge.

        The method simulates the network and returns the constraints and the input data for each edge.
        """
        constraints, edge_data = self.simulate(decisions, uncertain_params)
        for node, g in constraints.items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        return constraints, edge_data
        
    
    def evaluate_direct(self, decisions, uncertain_params):
        """
        Evaluates the network for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the nodes and the values are the constraints for each node.
            dict: A dictionary where the keys are the edges and the values are the input data for each edge.

        The method simulates the network and returns the constraints and the input data for each edge.
        """
        n_theta = [self.graph.nodes[node]['n_theta'] for node in self.graph.nodes]
        nu_pk = 0
        nu_pk_1 = 0
        n_d = 0
        aux_args = decisions[:, sum([self.graph.nodes[node]['n_design_args'] for node in self.graph.nodes]):]
        for node in self.graph.nodes:
            if not (uncertain_params.all() == None) :
                nu_pk = nu_pk_1 + n_theta[node]
                u_p = uncertain_params[:,nu_pk_1:nu_pk]
                if u_p.ndim == 1:
                    u_p = jnp.expand_dims(u_p, axis=1)

                nu_pk_1 = nu_pk


            if self.graph.in_degree()[node] == 0:
                if not (self.cfg.model.root_node_inputs[node] == 'None'):
                    inputs = jnp.array([self.cfg.model.root_node_inputs[node]]*decisions.shape[0])
                else:
                    inputs = jnp.empty((decisions.shape[0], u_p.shape[0], 0))
            else:
                inputs = jnp.concatenate([jnp.copy(self.graph.edges[predecessor, node]['input_data_store'])[:,:,:] for predecessor in self.graph.predecessors(node)], axis=-1)

            unit_nd = self.graph.nodes[node]['n_design_args']
            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, u_p)
            
            for successor in self.graph.successors(node):
                edge_data = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))
                if edge_data.ndim==2: edge_data = jnp.expand_dims(edge_data, axis=-1)
                self.graph.edges[node, successor]['input_data_store'] = edge_data

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node, constraint_type=self.type)

            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, outputs)


            n_d += unit_nd

        # constraint evaluation, information for extended KS bounds
        return {node: self.graph.nodes[node]['constraint_store'] for node in self.graph.nodes}, {edge: self.graph.edges[edge[0],edge[1]]['input_data_store'] for edge in self.graph.edges}
    

    def direct_evaluate(self, decisions, uncertain_params):
        """
        Evaluates the network for the given decisions and uncertain parameters.

        Args:
            decisions (array): Array of decisions.
            uncertain_params (array, optional): Array of uncertain parameters. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the nodes and the values are the constraints for each node.
            dict: A dictionary where the keys are the edges and the values are the input data for each edge.

        The method simulates the network and returns the constraints and the input data for each edge.
        """
        constraints, _ = self.evaluate_direct(decisions, uncertain_params)
        for node, g in constraints.items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]

        cons_ = jnp.concatenate([cons for cons in constraints.values()], axis=-1)

        return [cons_[i,:,:] for i in range(cons_.shape[0])]
 


class post_process_evaluation(network_simulator):
    """
    This class is responsible for post-processing the results of the network simulation.
    It extends the network_simulator class and provides additional functionality for post-processing.
    """
    def __init__(self, cfg, graph, constraint_evaluator):
        super().__init__(cfg, graph, constraint_evaluator, type_cons='post_process_evals')
        self.type = 'post_process_evals'

    def get_auxiliary_bounds(self):
        aux_bounds = self.cfg.case_study.KS_bounds.aux_args
        aux_lb = jnp.array([bound[0][0] for bound in aux_bounds])
        aux_ub = jnp.array([bound[0][1] for bound in aux_bounds])
        return aux_lb, aux_ub
    
    def wrap_get_constraints(self, solution):
        """
        Wraps the get_constraints method to handle the solution.

        Args:
            solution (array): The solution to be processed.

        Returns:
            Dataframe mapping the solution to the constraints.
            # NOTE set up custom for the current case study (haven't thought of a general way to do this yet)
        """
        logging.warning('This post-process evaluation is set up for a specific case study and is not generalised yet.')
        bounds = self.get_auxiliary_bounds()
        x_range = (bounds[0][0], bounds[1][0])
        y_range = (bounds[0][1], bounds[1][1])
        num_points = 200
        # Create a grid of points for the x and y axes
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        
        # Use numpy.meshgrid to create the 2D grid from the 1D arrays
        X, Y = np.meshgrid(x, y)

        # Flatten the X and Y grids into 1D arrays for the batch evaluation
        # This creates a "batch" of all coordinate pairs to be evaluated
        x_coords_batch = X.ravel().reshape(-1, 1)
        y_coords_batch = Y.ravel().reshape(-1, 1)

        # Combine the x and y coordinates into a single array of shape (num_points, 2)
        solution_batch = np.tile(solution, (num_points * num_points, solution.shape[0]))
        coords_batch = np.hstack((solution_batch, x_coords_batch, y_coords_batch, np.zeros((num_points*num_points,1))))

        uncertain_params = jnp.empty((num_points, 1))
        epsilon = self.get_constraints(coords_batch, uncertain_params)[5]
        df = {
            'x': np.reshape(x_coords_batch, X.shape),
            'y': np.reshape(y_coords_batch, X.shape),
            'z': np.reshape(epsilon.squeeze(), X.shape)
        }

        print(f'Post-process evaluation: {[v.shape for v in df.values()]} shape check.')

        return df
