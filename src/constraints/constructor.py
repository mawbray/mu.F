from abc import ABC
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit

from constraints.evaluator import process_constraint_evaluator, forward_constraint_evaluator, forward_constraint_decentralised_evaluator, backward_constraint_evaluator, backward_constraint_evaluator_general, forward_root_constraint_decentralised_evaluator, global_graph_upperlevel_NLP

class constraint_evaluator(ABC):
    def __init__(self, cfg, graph, node, pool=None, constraint_type='process'):
        self.cfg = cfg
        self.graph = graph
        self.node = node

        if constraint_type == 'process':
            self.constraint_evaluator = process_constraint_evaluator(cfg, graph, node, pool)
            self.evaluate = self.evaluate_process
        elif constraint_type == 'forward':
            self.constraint_evaluator = forward_constraint_evaluator(cfg, graph, node, pool)
            self.evaluate = self.evaluate_coupling
        elif constraint_type == 'forward_decentralized':
            self.constraint_evaluator = forward_constraint_decentralised_evaluator(cfg, graph, node, pool)
            self.evaluate = self.evaluate_coupling
        elif constraint_type == 'backward':
            if (len([prec for succ in self.graph.successors(node) for prec in self.graph.predecessors(succ) if prec > node])>0) and cfg.solvers.backward_coupling.tightening:
                self.constraint_evaluator = backward_constraint_evaluator_general(cfg, graph, node, cfg.solvers.evaluation_mode.forward) # use forward constraint pool in the general case/
                self.evaluate = self.evaluate_coupling
            else:
                self.constraint_evaluator = partial(backward_constraint_evaluator, cfg=cfg, graph=graph, node=node, pool=pool)
                self.evaluate = self.evaluate_coupling
        elif constraint_type == 'root_node_decentralized':
            self.constraint_evaluator = forward_root_constraint_decentralised_evaluator(cfg, graph, node, pool)
            self.evaluate = self.evaluate_coupling
        elif constraint_type == 'post_process_lower_level':
            self.constraint_evaluator = partial(backward_constraint_evaluator, cfg=cfg, graph=graph, node=node, pool=pool)
            self.evaluate = self.evaluate_coupling
        elif constraint_type == 'post_process_upper_level':
            self.constraint_evaluator = global_graph_upperlevel_NLP(cfg=cfg, graph=graph, node=node, pool=pool)
            self.evaluate = self.evaluate_global
        else:   
            raise ValueError('Invalid constraint type')

    def evaluate_process(self, design, inputs, aux, outputs):
        return self.constraint_evaluator(design, inputs, outputs, aux)
    
    def evaluate_coupling(self, inputs, aux):
        return self.constraint_evaluator(inputs, aux)
    
    def evaluate_global(self):
        return self.constraint_evaluator()

    
