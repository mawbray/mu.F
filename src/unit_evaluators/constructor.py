
from abc import ABC
from copy import copy
from functools import partial
import jax.numpy as jnp 
from jax import vmap, jit

from unit_evaluators.integrators import unit_dynamics
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

    def evaluate(self, design_args, input_args, uncertain_params=None):
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
        if dd_params.ndim<2: dd_params = jnp.expand_dims(dd_params, axis=1)
        sys_params = jnp.hstack([dd_params, design_args])


        return self.unit_cfg.evaluator(sys_params, input_args, uncertain_params)

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
        design_args, input_args = self.get_input_decision_split(decisions)
        if input_args.shape[1] == 0: 
            input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]]*design_args.shape[0])
        if uncertain_params is None:
            uncertain_params = jnp.empty((1,1))
        
        return self.evaluate(design_args, input_args, uncertain_params)
    
    def get_input_decision_split(self, decisions):
        n_d = self.graph.nodes[self.node]['n_design_args']
        design_args, input_args = decisions[:,:n_d], decisions[:,n_d:]
        return design_args, input_args


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
        self.n_theta = cfg.model.uncertain_parameters.n_theta[node]

        # if vmap is enabled in cfg, set the unit evaluation and decision dependent evaluation functions using vmap
        if cfg.case_study.vmap_evaluations:
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = vmap(vmap(jit(partial(unit_dynamics, cfg=cfg, node=node)), in_axes=(0,0, None), out_axes=0), in_axes=(None, None, 0), out_axes=1) # inputs are design args, input args, uncertain params
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation fn
            if graph.nodes[node]['unit_params_fn'] == 'Arrhenius':
                EA, R, A = jnp.array(cfg.model.arrhenius.EA[node]), jnp.array(cfg.model.arrhenius.R), jnp.array(cfg.model.arrhenius.A[node])
                self.decision_dependent_params = vmap(partial(arrhenius, Ea=EA, R=R, A=A), in_axes=(0, None), out_axes=0)
            elif graph.nodes[node]['unit_params_fn'] is None: # NOTE this allocation has not been tested
                self.decision_dependent_params = lambda x, y: jnp.empty(x.shape[0])
            else:
                raise NotImplementedError('Not implemented error')

        # if vmap is not enabled in cfg, set the unit evaluation and decision dependent evaluation functions without using vmap
        else: 
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = lambda x, y, z: jit(partial(unit_dynamics, cfg=cfg, node=node))(x.squeeze(), y.squeeze(), z.squeeze())
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation fn
            if graph.nodes[node]['unit_params_fn'] == 'Arrhenius':
                EA, R, A = jnp.array(cfg.model.arrhenius.EA[node]), jnp.array(cfg.model.arrhenius.R), jnp.array(cfg.model.arrhenius.A[node])
                self.decision_dependent_params = partial(arrhenius, Ea=EA, R=R, A=A)
            elif graph.nodes[node]['unit_params_fn'] is None:
                self.decision_dependent_params = lambda x, y: jnp.empty(x.shape[0]) # NOTE this allocation has not been testeds
            else:
                raise NotImplementedError('Not implemented error')

        return
    
class network_simulator(ABC):
    """
    Abstract base class for a network simulator.

    This class is responsible for simulating a network of interconnected nodes and edges. 
    Each node represents a unit operation in a process, and each edge represents the flow of material between units.
    """
    def __init__(self, cfg, graph, constraint_evaluator):
        self.cfg = cfg
        self.graph = graph.copy()
        self.constraint_evaluator = constraint_evaluator

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
        for node in self.graph.nodes:
            if not (uncertain_params == None) :
                u_p = uncertain_params[node]

            if self.graph.in_degree()[node] == 0:
                inputs = jnp.array([self.cfg.model.root_node_inputs[node]]*decisions.shape[0])
            else:
                inputs = jnp.hstack([jnp.copy(self.graph.edges[predecessor, node]['input_data_store']) for predecessor in self.graph.predecessors(node)])

            unit_nd = self.graph.nodes[node]['n_design_args']
            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions[:, n_d:n_d+unit_nd], inputs, u_p)
            
            for successor in self.graph.successors(node):
                self.graph.edges[node, successor]['input_data_store'] = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node)

            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, outputs)

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
        return constraints, edge_data
        