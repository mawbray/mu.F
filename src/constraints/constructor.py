from abc import ABC
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit

from evaluator import process_constraint_evaluator, forward_constraint_evaluator, backward_constraint_evaluator

class constraint_evaluator(ABC):
    def __init__(self, cfg, graph, node, pool, constraint_type):
        self.cfg = cfg
        self.graph = graph
        self.node = node

        if constraint_type == 'process':
            self.constraint_evaluator = process_constraint_evaluator(cfg, graph, node, pool)
        elif constraint_type == 'forward':
            self.constraint_evaluator = forward_constraint_evaluator(cfg, graph, node, pool)
        elif constraint_type == 'backward':
            self.constraint_evaluator = partial(backward_constraint_evaluator, cfg=cfg, graph=graph, node=node, pool=pool)

    def evaluate(self, iterated_object):
        return self.constraint_evaluator(iterated_object)
    
