
from abc import ABC
from functools import partial
import jax.numpy as jnp 
from jax import vmap, jit

from integrators import unit_dynamics
from utils import arrhenius_kinetics_fn as arrhenius
from src.constraints.evaluator import constraint_evaluator

class base_unit(ABC):
    def __init__(self, cfg, graph, node):
        self.cfg = cfg
        self.graph = graph
        self.node = node
        if self.graph.nodes[node]['unit_uncertainty'] :
            self.uncertainty = True

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
        if uncertain_params is None:
            uncertain_params = jnp.zeros((10,1))
        dd_params = self.get_decision_dependent_params(design_args, uncertain_params)
        sys_params = jnp.hstack([dd_params, design_args])

        if input_args is None: 
            inputs = jnp.array([self.cfg.root_node_inputs[self.node]]*design_args.shape[0])
        else: 
            inputs = input_args

        print(sys_params.shape, inputs.shape, uncertain_params.shape)

        return self.unit_cfg.evaluator(sys_params, inputs, uncertain_params)



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
        if cfg.vmap_unit_evaluation[self.node]:
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = vmap(vmap(jit(partial(unit_dynamics, cfg=cfg, node=node)), in_axes=(0,0, None), out_axes=0), in_axes=(None, None, 0), out_axes=1) # inputs are design args, input args, uncertain params
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation fn
            if graph.nodes[node]['unit_params_fn'] == 'Arrhenius':
                EA, R, A = cfg.arrhenius.EA[node], cfg.arrhenius.R, cfg.arrhenius.A[node]
                self.decision_dependent_params = vmap(vmap(partial(arrhenius, Ea=EA, R=R, A=A), in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)
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
                EA, R, A = cfg.arrhenius.EA[node], cfg.arrhenius.R, cfg.arrhenius.A[node]
                self.decision_dependent_params = partial(arrhenius, Ea=EA, R=R, A=A)
            elif graph.nodes[node]['unit_params_fn'] is None:
                self.decision_dependent_params = lambda x, y: jnp.empty(x.shape[0]) # NOTE this allocation has not been testeds
            else:
                raise NotImplementedError('Not implemented error')

        return
    
class network_simulator(ABC):
    def __init__(self, cfg, graph, constraint_evaluator):
        self.cfg = cfg
        self.graph = graph.copy()
        self.constraint_evaluator = constraint_evaluator

    def simulate(self, decisions, uncertain_params=None):
        
        for node in self.graph.nodes:
            if self.graph.nodes[node].in_degree == 0:
                inputs = None
            else:
                inputs = jnp.hstack([jnp.copy(self.graph.edges[predecessor, node]['input_data_store']) for predecessor in self.graph.predecessors(node)])

            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions, inputs, uncertain_params)
            
            for successor in self.graph.successors(node):
                self.graph.edges[node, successor]['input_data_store'] = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node)

            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(outputs)

        # constraint evaluation, information for extended KS bounds
        return {node: self.graph[node]['constraint_store'] for node in self.graph.nodes}, {edge: self.graph.edges[edge[0],edge[1]]['input_data_store'] for edge in self.graph.edges}

        