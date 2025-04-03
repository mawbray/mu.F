from abc import ABC
from typing import List, Tuple
from functools import partial

import pandas as pd
from visualisation.methods import init_plot, decompose_call, reconstruction_plot, design_space_plot

class visualiser(ABC):
    def __init__(self, cfg, G, data: pd.DataFrame=None, mode:str='forward', string:str='design_space', path=None):
        self.data = data
        self.cfg, self.G = cfg, G
        self.path = path
        assert string in ['initialisation', 'design_space', 'reconstruction', 'decomposition'], 'string must be one of "initialisastion", "design_space", "reconstruction", "decomposition" '
        if string == 'initialisation':
            self.visualiser =partial(init_plot, init=True, save=True)
        elif string == 'design_space':
            assert type(data) != type(None), 'design space plot requires data in the form of a dataframe'
            assert type(path) == type('hello'), 'design space plot requires a path to save to'
            self.visualiser = partial(design_space_plot, joint_data_direct=data, path=path)
        elif string == 'reconstruction':
            assert type(data) != type(None), 'reconstruction plot requires data in the form of a dataframe'
            self.visualiser = partial(reconstruction_plot, reconstructed_df=data, save=True, path=path)
        elif string == 'decomposition':
            if mode =='forward':
                self.visualiser = partial(decompose_call, init=False, path=path)
            elif mode == 'backward':
                self.visualiser = partial(decompose_call, init=True, path=path)
        

    def run(self):
        self.visualiser(self.cfg, self.G)
        return



if __name__ == '__main__':
    """ 
    plot of the bayesian optimization progress
    """ 

    import torch as t
    import numpy as np
    from methods import plotting_format
    import matplotlib.pyplot as plt

    root = './multirun/'
    date = '2025-01-14'
    time = '11-14-28'
    iteration = 0
    x_train = 'opt_x.pt'
    y_train = 'opt_y.pt'

    x = t.load(root + date + '/' + time + '/' + str(iteration) + '/' + x_train)
    y = t.load(root + date + '/' + time + '/' + str(iteration) + '/' + y_train)

    plotting_format()
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(list(range(1,11,1))), y, label='Evaluations', linewidth=2, marker='o', linestyle='--', markersize=10)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Objective value', fontsize=20)
    plt.xlim(1, 10)
    plt.legend()
    plt.savefig(root + date + '/' + time + '/' + str(iteration) + '/bo_progress.svg')

    