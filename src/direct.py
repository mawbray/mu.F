
import jax.numpy as jnp
from unit_evaluators.constructor import network_simulator
from samplers.constructor import construct_deus_problem_network
from constraints.constructor import constraint_evaluator
from samplers.utils import create_problem_description_deus_direct

from deus import DEUS

def apply_direct_method(cfg, graph):

    model = network_simulator(cfg, graph, constraint_evaluator)
    problem_description = create_problem_description_deus_direct(cfg, graph)
    solver =  construct_deus_problem_network(DEUS, problem_description, model)
    solver.solve()
    feasible_set, infeasible_set = solver.get_solution()
    for node in graph.nodes:
        graph.nodes[node]['fn_evals'] = model.function_evaluations[node]


    return feasible_set, infeasible_set