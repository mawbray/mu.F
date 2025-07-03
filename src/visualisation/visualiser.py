from abc import ABC
from typing import List, Tuple
from functools import partial
from copy import copy

import pandas as pd
from visualisation.methods import init_plot, decompose_call, polytope_plot, decomposition_plot, reconstruction_plot, design_space_plot, polytope_plot_2, design_space_plot_plus_polytope

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
        else:
            raise ValueError('string must be one of "initialisastion", "design_space", "reconstruction", "decomposition" ')
        

    def run(self):
        self.visualiser(self.cfg, self.G)
        return


def polytope_plot(pp, vertices, path=None):
    '''
    Plot the vertices of a polytope in 2D
    :param vertices: dict of vertices
    :return: None
    '''
    from scipy.spatial import ConvexHull
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    for i, j in indices:
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if (x_var, y_var) in list(vertices.keys()):
            points = np.array(vertices[(x_var, y_var)][0])
            if points.shape[0] < 3:
                continue  # Need at least 3 points for a polygon
            # Ensure the polygon is closed for plotting
            hull = ConvexHull(points)
            # Extract the vertices of the convex hull
            points = points[hull.vertices]
            # Unzip the points for plotting

            x, y = points[:, 0], points[:, 1]
            # Repeat the first point at the end to close the polygon
            # x_closed = np.append(x, x[0])
            # y_closed = np.append(y, y[0])
            ax.fill(x, y, alpha=0.7, color='blue', edgecolor='black', linewidth=1.5)
            ax.plot(x, y, color='black', linewidth=2, alpha=1)
    if path:
        plt.savefig(path, dpi=300)
    else:
        plt.savefig('centralised.png', dpi=300)
    return pp
    #



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
    from methods import hide_current_axis, polytope_plot_2
    plotting_format()

    def load_mat_file(file_path: str):
        '''
        Load a .mat file and return its contents as a dictionary.
        :param file_path: Path to the .mat file
        :return: Dictionary containing the .mat file data
        '''
        try:
            mat_data = loadmat(file_path)
            return mat_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The fi at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the .mat file: {e}")

    root = '../../paper_results/decentralised/matlab'
    backward = 'backwards_propagation_projection.mat'
    bf_tuned = 'tuneddecentralised_bf_propagation.mat'
    bf_untuned = 'untuneddecentralised_bf_propagation.mat'
    centralised = 'centralised_projections.mat'

    matlab_stem = [backward, bf_tuned, bf_untuned, centralised]
    paths = {stem: os.path.join(root, stem) for stem in matlab_stem}
    # Example usage:
    data = {}
    for stem, path in paths.items():
        try:
            data[stem] = load_mat_file(path)
            print(f"Loaded {stem} from {path}")
        except Exception as e:
            print(f"Failed to load {stem} from {path}: {e}")

    def get_pairwise_projections(projections_dict):
        """
        Given a dictionary where each key maps to a list of intervals (lower, upper),
        return a dictionary mapping (i, j) -> list of 2D rectangles (as tuples of (x_bounds, y_bounds)).
        """
        pairwise = {}
        keys = list(projections_dict.keys())

        for i, j in combinations(keys, 2):
            pairwise[(i, j)] = []
            # Assume each projections_dict[k] is a list of intervals (lower, upper)
            # and all lists are of the same length
            x_bounds = projections_dict[i][0]
            y_bounds = projections_dict[j][0]
            # Each box is defined by its lower and upper bounds in x and y
            x0, x1 = x_bounds
            y0, y1 = y_bounds
            # Vertices in order: (x0, y0), (x1, y0), (x1, y1), (x0, y1)
            vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pairwise[(i, j)].append(vertices)

        return pairwise | projections_dict
   
    print('done')
    backwards_projections = {}
    for k in range(5):
        backwards_projections[f'N{k+1}: P1'] = [x[k] for x in data[backward]['v_b_data_struct'][0]]
    
    tuned_bf_projections = {}
    for k in range(5):
        tuned_bf_projections[f'N{k+1}: P1'] = [x[k] for x in data[bf_tuned]['v_bf_data_struct'][0]]

    untuned_bf_projections = {}
    for k in range(5):
        untuned_bf_projections[f'N{k+1}: P1'] = [x[k] for x in data[bf_untuned]['v_bf_data_struct'][0]]
    
    # Convert the projections to a pairwise format
    #backwards_projections = get_pairwise_projections(backwards_projections)
    #tuned_bf_projections = get_pairwise_projections(tuned_bf_projections)
    #untuned_bf_projections = get_pairwise_projections(untuned_bf_projections

    centralised_projections = {}
    index = [x.dtype.names for x in data[centralised]['polytopes_by_pair']]
    for k, index in enumerate(index[0]):
        centralised_projections[('N'+index[-3]+ ': P1', 'N'+index[-1]+ ': P1')] = [x[k] for x in data[centralised]['polytopes_by_pair'][0]]

    df = {n: [-1.] for n in ['N1: P1', 'N2: P1', 'N3: P1', 'N4: P1', 'N5: P1'] }
    print(df)
    # Create a dummy DataFrame with those variables
    # We'll just fill in a few NaNs to get the right shape
    dummy_data = pd.DataFrame(df)
    print(dummy_data)
    DS_bounds = pd.DataFrame({
        'N1: P1': [-1.0, 1.0],
        'N2: P1': [-1.0, 1.0],
        'N3: P1': [-1.0, 1.0],
        'N4: P1': [-1.0, 1.0],
        'N5: P1': [-1.0, 1.0]
    })
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
            '''ax.fill_betweenx(
                y=[DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]],
                x1=DS_bounds[x_var].iloc[0],
                x2=DS_bounds[x_var].iloc[1],
                color='black',
                alpha=0.5
            )'''
            print(f'DS_bounds: {x_var}, {y_var}', (DS_bounds[x_var].iloc[0], DS_bounds[x_var].iloc[1]), (DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]))
    
    #pp = polytope_plot_2(pp, backwards_projections, color='red', path='backwards_projections.svg')
    #pp = polytope_plot(pp, centralised_projections, path = 'centralised_projectionsignore.svg')
    
    pp = polytope_plot_2(pp, untuned_bf_projections, color='green', path= 'untunedonly_bf_projections.svg')
    pp = polytope_plot_2(pp, tuned_bf_projections, color='red', path='untuned+tuned_bf_projections.svg')
    
    """
    root = 'multirun/'
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
    
    # TODO plot results from decomposition for affine study and sampling decompsoitons
    
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
    root    = 'multirun/2025-05-21/16-27-42/1'
    root2    = 'multirun/2025-05-21/16-27-42/0'
    config_path = os.path.join(root2, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem  = 'graph_backward-reconstructed_iterate_0.pickle'
    graph2_stem = 'graph_direct_complete.pickle'
    with open(os.path.join(root, graph_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph = pkl.load(file)
    with open(os.path.join(root2, graph2_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph2 = pkl.load(file)
    DS_bounds = get_ds_bounds(cfg, graph2)

    df = {n: [-1.] for node in cfg.case_study.process_space_names for n in node}
    print(df)
    # Create a dummy DataFrame with those variables
    # We'll just fill in a few NaNs to get the right shape
    dummy_data = pd.DataFrame(df)
    print(dummy_data)
    joint_set= pd.DataFrame({col: graph2.graph['feasible_set'][0][:,i] for i,col in enumerate(cfg.case_study.design_space_dimensions)})#pd.read_excel(os.path.join(root, "inside_samples_backward_iterate_0.xlsx"), index_col=0) #
    joint_set.columns = [col.replace('N5: G', 'G: ') for col in joint_set.columns]
    DS_bounds.columns = [col.replace('N5: G', 'G: ') for col in DS_bounds.columns]
    print(joint_set)

    fig = plt.figure(figsize=(35, 35))
    pp = sns.PairGrid(joint_set, vars=joint_set.columns, aspect=1.4)
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
            '''ax.fill_betweenx(
                y=[DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]],
                x1=DS_bounds[x_var].iloc[0],
                x2=DS_bounds[x_var].iloc[1],
                color='black',
                alpha=0.5
            )'''
            print(f'DS_bounds: {x_var}, {y_var}', (DS_bounds[x_var].iloc[0], DS_bounds[x_var].iloc[1]), (DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]))

    #pp = polytope_plot_2(pp, vb)
    #pp = decomposition_plot(cfg, graph, pp, save=False, path='decomposed_pair_grid_plot_F')
    #pp = design_space_plot(cfg, graph, joint_set, path='direct_grid_re_plot')
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    for  i,j in indices:
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in joint_set.columns and y_var in joint_set.columns:
            #ax.set_xlim(DS_bounds[x_var].iloc[0], DS_bounds[x_var].iloc[1])
            #ax.set_ylim(DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1])
            ax.scatter(
                joint_set[x_var],
                joint_set[y_var],
                edgecolor="k",
                c="b", linewidth=0.5,)
            ax.set_xticks([DS_bounds[x_var].iloc[0], DS_bounds[x_var].iloc[1]])
            ax.set_yticks([DS_bounds[y_var].iloc[0], DS_bounds[y_var].iloc[1]])

    for i, x_var in enumerate(pp.x_vars): 
        for j, y_var in enumerate(pp.y_vars):
            if x_var == y_var:
                print(f"Removing axis at {i}, {j} for variable {x_var} as it is the same as {y_var}")
                pp.axes[i,j].set_visible(False)
        

    #pp = polytope_plot_2(pp, vb)
    pp.savefig("direct_grid_re_plot.png", dpi=300)
    """
   
    """
    root    = '../paper_results/centralised/1#/matlab/combined_results'
    root2   = '../paper_results/centralised/1#/forward/0'
    config_path = os.path.join(root2, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem  = 'graph_direct_complete.pickle'
    matlab_stem = 'simultaneous_projections.mat'
    graph2_stem = 'graph_forward-reconstructed_iterate_0.pickle'
    config_path = os.path.join(root2, '.hydra', 'config.yaml')
    cfg2 = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg2, False)
    
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

    joint_set =  pd.read_excel(os.path.join(root2, "inside_samples_forward_iterate_0.xlsx"), index_col=0) #{col: graph.graph['feasible_set'][0][:,i] for i,col in enumerate(cfg2.case_study.design_space_dimensions)}
    joint_set = pd.DataFrame(joint_set)
    print(joint_set)
    
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

    # the ultimate plot
    pp = init_plot(cfg2, graph2, pp= pp, init=False, save=False)
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
    pp = polytope_plot(pp, s_p)
    #pp.map_lower(sns.scatterplot, data=joint_set, edgecolor="k", c="b",  linewidth=0.5)
    pp.savefig(os.path.join(root2, "reconstructed_with_polytope_pair_grid_plot.svg"), dpi=10)
    polytope_plot(sim_projections, path='vertices_all')

    polytope_plot(vertices_sim, path='vertices_sim')
    polytope_plot(vertices_forwardbackward, path='vertices_forwardbackward')
    polytope_plot(vertices_forward, path='vertices_forward')
    polytope_plot(vertices_backward, path='vertices_backward')
    """
    