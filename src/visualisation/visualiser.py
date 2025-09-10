from abc import ABC
from typing import List, Tuple
from functools import partial

import pandas as pd
from visualisation.methods import init_plot, decompose_call, polytope_plot, decomposition_plot, reconstruction_plot, design_space_plot, polytope_plot_2, design_space_plot_plus_polytope, post_process_upper_solution

class visualiser(ABC):
    def __init__(self, cfg, G, data: pd.DataFrame=None, mode:str='forward', string:str='design_space', path=None):
        self.data = data
        self.cfg, self.G = cfg, G
        self.path = path
        assert string in ['initialisation', 'design_space', 'reconstruction', 'decomposition', 'post_process_upper'], 'string must be one of "initialisastion", "design_space", "reconstruction", "decomposition", "post_process_upper" '
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
        elif string == 'post_process_upper':
            self.visualiser = partial(post_process_upper_solution, solution=data)
        else:
            raise ValueError('string must be one of "initialisastion", "design_space", "reconstruction", "decomposition", "post_process_upper"')


    def run(self):
        self.visualiser(self.cfg, self.G)
        return
    
    def run_with_args(self,*args):
        """
        This method is used to run the visualiser with additional arguments.
        It allows for flexibility in how the visualiser is executed.
        """
        return self.visualiser(*args)


"""def polytope_plot(pp, vertices, path=None):
    
    Plot the vertices of a polytope in 2D
    :param vertices: dict of vertices
    :return: None
    
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from methods import plotting_format
    plotting_format()
    
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

    return pp"""
    #plt.savefig(path + '.svg', dpi=300)



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
    from methods import plotting_format, get_ds_bounds, hide_current_axis, initializer_cp
    import seaborn as sns
    from itertools import combinations
    import glob
    plotting_format()

    """root = 'multirun/'
    date = '2025-05-16'
    time = '19-42-10'
    iteration = 0
    x_train = 'opt_x.pt'
    y_train = 'opt_y.pt'
    graph0 = "graph_backward_iterate_0_node_0.pickle"
    graph10 = "graph_backward-forward_iterate_10.pickle"

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

    # root
    root= 'src/matlab_results/decentralised'
    bp = 'backwards_propagation_projection.mat'
    cp = 'centralised_projections.mat'
    dp = 'decentralised_bf_propagation.mat'

    # load the .mat files
    bp_data = loadmat(os.path.join(root, bp))
    cp_data = loadmat(os.path.join(root, cp))
    dp_data = loadmat(os.path.join(root, dp))

    print("Backwards Propagation Projections:")

    # TODO plot results from decomposition for affine study and sampling decompsoitons
    """
    vertices_sim = {0: [(1,-0.501604), (1,-1), (0.543809, -1)], 1: [(-0.671903,-1), (0.343159,-1), (1, -0.326864), (1,0.713381)], 
                    2: [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], 3: [(-1,-1), (-0.160083,-1), (1,0.713775), (1,1), (-1,1)], 4: [(-0.895464,-1), (1,-1), (1,1), (0.0384752,1)]}
    vertices_forwardbackward = vertices_sim.copy()

    vertices_forward = {0: [(1,-0.501604), (1,-1), (0.543809, -1)], 1: [(-0.671903,-1), (1,-1), (1,0.713381)],
                        2: [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], 3: [(-1,-1), (-0.160083,-1), (1,0.713775), (1,1), (-1,1)], 4: [(-0.895464,-1), (1,-1), (1,1), (0.0384752,1)]}
    
    vertices_backward = {0: [(1,-0.501604), (1,-1), (0.543809, -1)], 1: [(-0.671903,-1), (0.343159,-1), (1, -0.326864), (1,0.713381)],
                        2: [(-1,-1), (-0.531559,-1), (0.531537,1), (-1,1)], 3: [(-1,-1), (-1,1), (1,1), (1,-1)], 4: [(-1,-1), (1,-1), (1,1), (-0.44417,1), (-1, -0.190292)]}
    

    vf = {f'N{i+1}: P{j+1}': [v[j] for v in vertices_forward[i]] for i in range(len(vertices_forward)) for j in range(len(vertices_forward[i][0]))}
    vb = {f'N{i+1}: P{j+1}': [v[j] for v in vertices_backward[i]] for i in range(len(vertices_backward)) for j in range(len(vertices_backward[i][0]))}
    vs = {f'N{i+1}: P{j+1}': [v[j] for v in vertices_sim[i]] for i in range(len(vertices_sim)) for j in range(len(vertices_sim[i][0]))}
    

    # Replot the design space for introductory study
    root    = 'multirun/2025-04-08/17-09-01/0'
    config_path = os.path.join(root, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem  = 'graph_forward-reconstructed_iterate_0.pickle'
    
    with open(os.path.join(root, graph_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph = pkl.load(file)

    DS_bounds = get_ds_bounds(cfg, graph)

    df = {n: [-1.] for node in cfg.case_study.process_space_names for n in node}
    print(df)
    # Create a dummy DataFrame with those variables
    # We'll just fill in a few NaNs to get the right shape
    dummy_data = pd.DataFrame(df)
    print(dummy_data)
    
    fig = plt.figure(figsize=(35, 35))
    pp = sns.PairGrid(dummy_data, vars=dummy_data.columns, aspect=1.4)
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)
    # Do nothing to populate — you now have an empty grid
    
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')

    pp = polytope_plot_2(pp, vb)
    pp.savefig("vertices_backward.svg", dpi=300)

    #pp = design_space_plot(cfg, graph, pp, save=True, path='decomposed_pair_grid_plot')
    # pp = decomposition_plot(cfg, graph, pp, save=True, path='decomposed_pair_grid_plot_F')
    """
    
    
    def load_mat_file(file_path: str):
        '''
        Load a .mat file and return its contents as a dictionary.
        :param file_path: Path to the .mat file
        :return: Dictionary containing the .mat file data
        '''
        try:
            mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
            return mat_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The fi at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the .mat file: {e}")

    """root    = 'multirun/2025-04-30/16-03-22/1'
    root2   = 'multirun/2025-04-30/16-03-22/1'
    config_path = os.path.join(root, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem  = 'graph_direct_complete.pickle'
    matlab_stem = 'simultaneous_projections.mat'
    graph2_stem = 'graph_backward-reconstructed_iterate_0.pickle'
    config_path = os.path.join(root2, '.hydra', 'config.yaml')
    cfg2 = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg2, False)"
    
    with open(os.path.join(root, graph_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph = pkl.load(file)

    with open(os.path.join(root2, graph2_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph2 = pkl.load(file)

    graph2 = graph2 
    dims = [3,2,2]

    #columns = [f'N{j+1}: P{i+1}' for j,n in enumerate(dims) for i in range(n)]
    
    #print(columns)

    joint_set2 =  pd.read_excel(os.path.join(root2, "inside_samples_backward_iterate_0.xlsx"), index_col=0) #{col: graph.graph['feasible_set'][0][:,i] for i,col in enumerate(cfg2.case_study.design_space_dimensions)}
    joint_set2 = pd.DataFrame(joint_set2)
    print(joint_set2)
    #
    # Rename all columns with 'N5: G' to start with 'G'
    #joint_set2.columns = columns

    #print([[f'N{j+1}: P{i+1}' for i,col_ in enumerate(col)] for j,col in enumerate(cfg.case_study.process_space_names)])
    #OmegaConf.update(cfg2, 'case_study.process_space_names', [[f'N{j+1}: P{i+1}' for i,col_ in enumerate(col)] for j,col in enumerate(cfg.case_study.process_space_names)])
    #OmegaConf.update(cfg2, 'case_study.design_space_dimensions', columns)
    
    print(cfg2.case_study.process_space_names)
    print(cfg2.case_study.design_space_dimensions)

    print(joint_set2)

    
    fig = plt.figure(figsize=(35, 35))
    pp = sns.PairGrid(joint_set2[joint_set2.columns],aspect=1.4)
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)

    indices = zip(*np.tril_indices_from(pp.axes, -1))

     for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')
    print([graph2.nodes[node]['extendedDS_bounds'] for node in graph2.nodes])
    #pp = decompose_call(cfg2, graph2, path='decomposed_pair_grid_plot', init=False)
    
    init_df = graph2.graph['initial_forward_pass']
    #init_df.columns = [col.replace('N5: G', 'G: ') for col in init_df.columns]
    #print(init_df)
    
    DS_bounds = get_ds_bounds(cfg2, graph2)
    #DS_bounds.columns = [col.replace('N5: G', 'G: ') for col in DS_bounds.columns]
    print(DS_bounds)
    #joint_set2.columns = [col.replace('N5: G', 'G: ') for col in joint_set2.columns]
    print(joint_set2)
    
    pp = initializer_cp(joint_set2)
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.set_xticks([DS_bounds[x_var].iloc[0], DS_bounds[x_var].iloc[1]])
            ax.set_yticks([DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]])
            sns.scatterplot(x=x_var, y=y_var, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5, ax=ax)
    
    pp = decomposition_plot(cfg2, graph2, pp, save=False, path='decomposed_pair_grid_plot')
    pp.map_lower(sns.scatterplot, data=joint_set2, edgecolor="k", c="b",  linewidth=0.5)
    pp.savefig(os.path.join(root, "reconstruction_pair_grid_plot_replot.svg"), dpi=100)

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
    

    root = 'matlab_results/decentralised'
    # Load all files in the root directory
    file_paths = glob.glob(os.path.join(root, '*'))

    # Print the loaded file paths for verification
    print("Loaded files:")
    x = ['backwards', 'centralised', 'decentralised']
    for file_path in file_paths:
        load_mat_file(file_path)


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
    pp.map_diag(hide_current_axis)
    #pp.map_lower(sns.scatterplot, data=joint_set, edgecolor="k", c="b",  linewidth=0.5)
    pp.savefig(os.path.join(root2, "reconstructed_with_polytope_pair_grid_plot.svg"), dpi=20)
    polytope_plot(sim_projections, path='vertices_all')

    polytope_plot(vertices_sim, path='vertices_sim')
    polytope_plot(vertices_forwardbackward, path='vertices_forwardbackward')
    polytope_plot(vertices_forward, path='vertices_forward')
    polytope_plot(vertices_backward, path='vertices_backward')
    
    """