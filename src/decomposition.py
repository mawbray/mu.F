
from functools import partial
import pandas as pd
from integration import apply_decomposition
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from unit_evaluators.constructor import network_simulator
from constraints.constructor import constraint_evaluator
from samplers.space_filling import sobol_sampler
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
            operations = {0: partial(apply_decomposition, precedence_order=self.precedence_order, mode=m, max_devices=self.max_devices, total_iterations=self.total_iterations)}
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
    
    def run(self):

        for i in range(len(self.mode)):
            operations, visualisations = self.define_operations(i)
            for key in operations.keys():
                self.G = operations[key](self.cfg, self.G).run()
                visualisations[key](self.cfg, self.G).run()
                save_graph(self.G.copy(), self.mode[i] + '_iterate_' + str(i))
            if self.cfg.reconstruction.reconstruct[i]:
                self.reconstruct(self.mode[i], i)
            self.update_precedence_order(self.mode[i])
        

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
        visualiser(self.cfg, self.G, df, 'reconstruction', path=f'reconstruction_{m}_iterate_{i}').visualise()
        df.to_excel(f'inside_samples_{m}_iterate_{i}.xlsx')
        save_graph(self.G.copy(), m + '-reconstructed'+ '_iterate_' + str(i))

        return
    
    def update_precedence_order(self, m):
        precedence_order = self.original_precedence_order.copy()

        if m == 'backward' or 'forward-backward':
            for node in self.G.nodes():
                if self.G.in_degree(node) == 0:
                    precedence_order.remove(node)
        elif m == 'forward' or 'backward-forward':
            for node in self.G.nodes():
                if self.G.out_degree(node) == 0:
                    precedence_order.remove(node)

        self.precedence_order = precedence_order

        return
        

