from abc import ABC
from typing import List, Tuple
from functools import partial

import pandas as pd
from visualisation.methods import init_plot, decompose_call, reconstruction_plot, design_space_plot

class visualiser(ABC):
    def __init__(self, cfg, G, data: pd.DataFrame=None, string:str='design_space', path=None):
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
            self.visualiser = partial(decompose_call, path=path)
        

    def visualise(self):
        self.visualiser(self.cfg, self.G)
        return



