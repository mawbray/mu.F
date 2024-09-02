
import jax.numpy as jnp
import jax.random as random
from unit_evaluators.constructor import network_simulator
from samplers.constructor import construct_deus_problem_network
from constraints.constructor import constraint_evaluator
from samplers.utils import create_problem_description_deus_direct, get_network_uncertain_params
from cs_assembly import case_study_constructor
from visualisation.visualiser import visualiser
from deus import DEUS
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle 
import itertools


def apply_direct_method(cfg, graph):

    model = network_simulator(cfg, graph, constraint_evaluator)
    problem_description = create_problem_description_deus_direct(cfg, graph)
    solver =  construct_deus_problem_network(DEUS, problem_description, model)
    solver.solve()
    feasible_set, infeasible_set = solver.get_solution()
    for node in graph.nodes:
        graph.nodes[node]['fn_evals'] = model.function_evaluations


    return feasible_set, infeasible_set

@hydra.main(config_path="config", config_name="integrator")
def direct_evaluation(cfg: DictConfig) -> None:


    # Load the identified sets
    stems = ['/home/mmowbray/Github/feasibility/mu.F/src/multirun/2024-08-28/15-39-48/0/']
    leaves = ['graph_forward_iterate_1.pickle']

    with open(stems[0] + leaves[0], 'rb') as f:
        G = pickle.load(f)

    sets = [G.nodes[node]['live_set_inner'] for node in G.nodes]

    max_value = max(sets[0].shape)
    random_vector = random.randint(key=random.PRNGKey(0), shape=(10000,), minval=0, maxval=max_value)

    joint_live_set = jnp.hstack([sets[i][random_vector, :] for i in range(len(sets))])

    # rebuild the case study graph 
    # Set the maximum number of devices

    # Construct the case study graph
    G = case_study_constructor(cfg)   # TODO integration of case study construction G is a networkx graph - need to update case study contructor

    # Save the graph to a file

    model = network_simulator(cfg, G, constraint_evaluator)
    _, _, up = get_network_uncertain_params(cfg)
    constraint_evaluations = model.direct_evaluate(joint_live_set, up[0]['c'].reshape(1, -1))
    constraint_evaluations = [jnp.vstack(constraint_evaluations)]
    feasible = []
    for cons_g in constraint_evaluations:
        max_g = jnp.min(cons_g, axis=1)
        x = jnp.where(max_g >= 0, 1, 0).squeeze()
        feasible_live_set = joint_live_set[x == 1, :]
        feasible.append(feasible_live_set)


    df = pd.DataFrame({key: feasible_live_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
    visualiser(cfg, G, data=df, string='design_space', path=f'reconstruction').visualise()
    

    return feasible


if __name__ == '__main__':

    direct_evaluation()
    