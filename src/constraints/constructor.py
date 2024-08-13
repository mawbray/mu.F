from abc import ABC
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit

from constraints.evaluator import process_constraint_evaluator, forward_constraint_evaluator, backward_constraint_evaluator

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
            self.evaluate = self.evaluate_forward
        elif constraint_type == 'backward':
            self.constraint_evaluator = partial(backward_constraint_evaluator, cfg=cfg, graph=graph, node=node, pool=pool)
            self.evaluate = self.evaluate_backward
        else:   
            raise ValueError('Invalid constraint type')

    def evaluate_process(self, design, inputs, outputs):
        return self.constraint_evaluator(outputs)
    
    def evaluate_forward(self, inputs):
        return self.constraint_evaluator(inputs)
    
    def evaluate_backward(self, outputs):
        return self.constraint_evaluator(outputs)
    
