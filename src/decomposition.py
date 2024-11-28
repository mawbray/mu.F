
from functools import partial
import pandas as pd
import networkx as nx
import jax 
import logging

from integration import apply_decomposition
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from unit_evaluators.constructor import network_simulator
from constraints.constructor import constraint_evaluator
from samplers.space_filling import sobol_sampler
from samplers.algorithms.bo.alg import bayesian_optimization
from samplers.appproximators import calculate_box_outer_approximation  
from visualisation.visualiser import visualiser
from utils import *

class decomposition:
    def __init__(self, cfg, G, precedence_order, mode='forward', max_devices=1):
        self.cfg = cfg
        self.G = G
        self.original_precedence_order = precedence_order.copy()
        self.precedence_order = precedence_order.copy()
        self.total_iterations = len(mode)
        self.mode = mode
        self.max_devices = max_devices
        self.define_sampler()
        self.define_approximator()


    def define_sampler(self):
        if self.cfg.init.sampler == 'sobol':
            self.sampler = sobol_sampler()
        return 
    
    def define_approximator(self):
        if self.cfg.samplers.ku_approximation == 'box':
            self.approximator = calculate_box_outer_approximation
        return

    def define_operations(self, iteration):
        m = self.mode[iteration]
        if m == 'forward' or m == 'backward-forward':
            operations, visualisations = {}, {}
            k = len(operations)
            operations[k] = partial(apply_decomposition, precedence_order=self.precedence_order, mode=m, max_devices=self.max_devices)
            visualisations[k] = partial(visualiser, string='decomposition', path=f'decomposition_{m}_iterate_{iteration}')
        elif m == 'backward' or m == 'forward_backward':
            operations, visualisations = {}, {}
            if iteration == 0:
                operations[0] = partial(initialisation, network_simulator=network_simulator, constraint_evaluator=constraint_evaluator, sampler=self.sampler, approximator=self.approximator)
                visualisations[0] = partial(visualiser, string='initialisation', path=f'initialisation_{m}_iterate_{iteration}')
            k = len(operations)
            operations[k] = partial(apply_decomposition, precedence_order=self.precedence_order, mode=m, max_devices=self.max_devices)
            visualisations[k] = partial(visualiser, string='decomposition', path=f'decomposition_{m}_iterate_{iteration}')

        return operations, visualisations
    
    def run(self, iterations=0):

        for i in range(len(self.mode)):
            operations, visualisations = self.define_operations(i)
            for key in operations.keys():
                self.G = operations[key](self.cfg, self.G).run()
                visualisations[key](self.cfg, self.G).run()
                save_graph(self.G.copy(), self.mode[i] + '_iterate_' + str(i+iterations))
            if self.cfg.reconstruction.reconstruct[i]:
                self.reconstruct(self.mode[i], i+iterations)
            self.update_precedence_order(self.mode[i])

        return self.G
        

    def reconstruct(self, m, i):

        network_model = network_simulator(self.cfg, self.G, constraint_evaluator)
        joint_live_set, joint_live_set_prob = reconstruction(self.cfg, self.G, network_model).run() # TODO update uncertainty evaluations
        # update the graph with the function evaluations
        for node in self.G.nodes():
            self.G.nodes[node]["fn_evals"] += network_model.function_evaluations[node]

        # visualisation of reconstruction
        if self.cfg.reconstruction.plot_reconstruction == 'nominal_map':
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(self.cfg.case_study.design_space_dimensions)})
        elif self.cfg.reconstruction.plot_reconstruction == 'probability_map':
            df = pd.DataFrame({key: joint_live_set[:,i] for i, key in enumerate(self.cfg.case_study.design_space_dimensions)})
            df['probability'] = joint_live_set_prob
        visualiser(self.cfg, self.G, df, 'reconstruction', path=f'reconstruction_{m}_iterate_{i}').run()
        df.to_excel(f'inside_samples_{m}_iterate_{i}.xlsx')
        save_graph(self.G.copy(), m + '-reconstructed'+ '_iterate_' + str(i))

        return
    
    def update_precedence_order(self, m):
        precedence_order = self.original_precedence_order.copy()
        if self.cfg.method == 'decomposition':
            if m == 'backward' or 'forward-backward':
                for node in self.G.nodes():
                    if self.G.in_degree(node) == 0:
                        precedence_order.remove(node)
            elif m == 'forward' or 'backward-forward':
                for node in self.G.nodes():
                    if self.G.out_degree(node) == 0:
                        precedence_order.remove(node)
        else: pass
        self.precedence_order = precedence_order

        return
        


def update_constraint_tuning_parameters(G, xi):
    """
    Update the constraint tuning parameters in the graph.
    """

    for k, node in enumerate(G.nodes()):
        xi_input  = jnp.array(xi[k]).squeeze()
        G.nodes[node]['constraint_backoff'] = xi_input

    return G


def run_a_single_evaluation(xi, cfg, G):
    # Set the maximum number of devices
    max_devices = len(jax.devices('cpu'))   

    # update the constraint parmeters.
    G = update_constraint_tuning_parameters(G, xi)

    # getting precedence order
    precedence_order = list(nx.topological_sort(G))
    m = ['backward-forward']
    # run the decomposition
    G = decomposition(cfg, G, precedence_order, m, max_devices).run()
    G.graph['iterate'] += 1

    return -sum([G.nodes[node]['log_evidence'] for node in G.nodes()]) # minimise negative log evidence


def decomposition_constraint_tuner(cfg, G, max_devices):
    # getting precedence order
    precedence_order = list(nx.topological_sort(G))
    # run the backward decomposition
    G = decomposition(cfg, G, precedence_order, ['backward'], max_devices).run()

    # build up to the forward pass for BO
    G_init = G.copy()
    G.graph['iterate'] = 1

    fn = partial(run_a_single_evaluation, cfg=cfg, G=G) 
    
    lower_bound = [cfg.samplers.bo.bounds.min]*len(G.nodes())
    upper_bound = [cfg.samplers.bo.bounds.max]*len(G.nodes())
    num_initial_points = cfg.samplers.bo.num_initial_points
    num_iterations = cfg.samplers.bo.num_iterations

    xi_opt, best_index = bayesian_optimization(fn, lower_bound, upper_bound, num_initial_points, num_iterations)


    logging.info("------- Finished -------")
    logging.info("Best candidate: {}".format(xi_opt))
    logging.info("Best index: {}".format(best_index + 1))
    logging.info("------------------------")
