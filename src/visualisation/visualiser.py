from abc import ABC
from typing import List, Tuple
from functools import partial

import pandas as pd
from visualisation.methods import init_plot, decompose_call, polytope_plot, decomposition_plot, reconstruction_plot, design_space_plot, design_space_plot_plus_polytope

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


'''def polytope_plot(vertices, path=None):
    """
    Plot the vertices of a polytope in 2D
    :param vertices: dict of vertices
    :return: None
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from methods import plotting_format
    plotting_format()


    fig, ax = plt.subplots(1, len(list(vertices.keys())), figsize=(4 * len(vertices), 4))
    
    for i, (key, vertex) in enumerate(vertices.items()):
        x = [v[0] for v in vertex]
        y = [v[1] for v in vertex]
        points = np.hstack([np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)]).reshape(-1, 2)
        hull = ConvexHull(points)

        # Hull vertices come back in a certain order, which we can use directly:
        hull_vertices = points[hull.vertices]

        # Unzip for plotting
        x_hull, y_hull = zip(*hull_vertices)

        ax[i].fill(x_hull, y_hull, alpha=0.9, color='red', edgecolor='black', linewidth=1.5)
        ax[i].set_xlabel(f'N{i+1}: P1')
        ax[i].set_ylabel(f'N{i+1}: P2')
        #ax[i].legend()
        ax[i].set_xlim(-1,1)
        ax[i].set_ylim(-1,1)
        ax[i].set_xticks(np.arange(-1, 1.1, 0.5))
        ax[i].set_yticks(np.arange(-1, 1.1, 0.5))
        ax[i].set_title(f'Node {i+1}', fontsize=15)
    plt.tight_layout()

    plt.savefig(path + '.svg', dpi=300)'''



if __name__ == '__main__':
    """ 
    plot of the bayesian optimization progress
    """ 

    import torch as t
    import numpy as np
    from methods import plotting_format
    import matplotlib.pyplot as plt
    import dill as pkl
    import os
    from scipy.io import loadmat
    import sys
    from omegaconf import OmegaConf
    from scipy.spatial import ConvexHull
    from methods import plotting_format
    import seaborn as sns
    from itertools import combinations
    plotting_format()
    """root = './multirun/'
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
    """

    
    vertices_sim = {(0,1): [(1,-0.501604), (1,-1), (0.543809, -1)], (2,3): [(-0.671903,-1), (0.343159,-1), (1, -0.326864), (1,0.713381)], 
                    (4,5): [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], (6,7): [(-1,-1), (-0.160083,-1), (1,0.713775), (1,1), (-1,1)], (8,9): [(-0.895464,-1), (1,-1), (1,1), (0.0384752,1)]}

    def load_mat_file(file_path: str):
        """
        Load a .mat file and return its contents as a dictionary.
        :param file_path: Path to the .mat file
        :return: Dictionary containing the .mat file data
        """
        try:
            mat_data = loadmat(file_path)
            return mat_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The fi at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the .mat file: {e}")

    root    = 'multirun/2025-04-04/11-49-28/0'
    root2   = 'multirun/2025-04-08/17-09-01/0'
    config_path = os.path.join(root, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem  = 'graph_direct_complete.pickle'
    matlab_stem = 'simultaneous_projections.mat'
    graph2_stem = 'graph_forward-reconstructed_iterate_0.pickle'
    config_path = os.path.join(root2, '.hydra', 'config.yaml')
    cfg2 = OmegaConf.load(config_path)
    joint_set = pd.read_excel(os.path.join(root2, 'inside_samples_forward_iterate_0.xlsx'), index_col=0)
    print(joint_set)
    
    
    with open(os.path.join(root, graph_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph = pkl.load(file)

    with open(os.path.join(root2, graph2_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph2 = pkl.load(file)


    # Load the .mat file
    sim_projections = load_mat_file(os.path.join(root, matlab_stem))

    # Extract the 'sim_projections' variable from the loaded dictionary
    s_p = {}
    
    comv = combinations(range(1,11), 2)

    # Extract the 'sim_projections' variable from the loaded dictionary
    sim_p = [sim_proj for key, sim_proj in sim_projections.items() if isinstance(sim_proj, np.ndarray)]
    keys = cfg2.case_study.design_space_dimensions
    key_c = {}
    for i, key in enumerate(keys):
        key_c[i+1] = key

    print(key_c)

    print(sim_p)

    for i, j in zip(comv,sim_p[0][0][0]): 
        x = [v[0] for v in j]
        y = [v[1] for v in j]
        points = np.hstack([np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)]).reshape(-1, 2)
        hull = ConvexHull(points)
        # Hull vertices come back in a certain order, which we can use directly:
        hull_vertices = points[hull.vertices]
        # Unzip for plotting
        x_hull, y_hull = zip(*hull_vertices)
        s_p[(key_c[i[0]], key_c[i[1]])] = (x_hull, y_hull)
        

    print(s_p)
    # the ultimate plot
    pp = init_plot(cfg2, graph2, pp= None, init=False, save=False)
    pp = polytope_plot(pp, s_p)
    pp = decomposition_plot(cfg, graph2, pp, save=False, path='decomposed_pair_grid_plot')
    # Get the indices of the lower triangle of the pair grid plot
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in joint_set.columns and y_var in joint_set.columns:
            print('yes')
            sns.scatterplot(x=x_var, y=y_var, data=joint_set[[x_var,y_var]], edgecolor="k", c='blue', alpha=0.8, ax=ax)
        
    #pp.map_lower(sns.scatterplot, data=joint_set, edgecolor="k", c="b",  linewidth=0.5)
    pp.savefig(os.path.join(root2, "reconstructed_with_polytope_pair_grid_plot.svg"), dpi=300)





    vertices_forwardbackward = vertices_sim.copy()

    vertices_forward = {0: [(1,-0.501604), (1,-1), (0.543809, -1)], 1: [(-0.671903,-1), (1,-1), (1,0.713381)],
                        2: [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], 3: [(-1,-1), (-0.160083,-1), (1,0.713775), (1,1), (-1,1)], 4: [(-0.895464,-1), (1,-1), (1,1), (0.0384752,1)]}
    
    vertices_backward = {0: [(1,-0.501604), (1,-1), (0.543809, -1)], 1: [(-0.671903,-1), (0.343159,-1), (1, -0.326864), (1,0.713381)],
                        2: [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], 3: [(-1,-1), (-1,1), (1,1), (1,-1)], 4: [(-1,-1), (1,-1), (1,1), (-0.44417,1), (-1, -0.190292)]}


    """polytope_plot(sim_projections, path='vertices_all')

    polytope_plot(vertices_sim, path='vertices_sim')
    polytope_plot(vertices_forwardbackward, path='vertices_forwardbackward')
    polytope_plot(vertices_forward, path='vertices_forward')
    polytope_plot(vertices_backward, path='vertices_backward')
    """