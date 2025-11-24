import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import logging

DEFAULT_DPI = 20

def plotting_format():
    font = {"family": "serif", "weight": "bold", "size": 20}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rc("axes", labelsize=25)  # fontsize of the x and y label
    plt.rc("axes", linewidth=3)
    plt.rc("axes", labelpad=30)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)

    return


def initializer_cp(df):
    plotting_format()
    fig = plt.figure(figsize=(35, 35))
    cols = df.columns
    pp = sns.PairGrid(
        df[cols],
        aspect=1.4
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)
    
    return pp

def get_ds_bounds(cfg, G):
    DS_bounds = [np.array(G.nodes[unit_index]['KS_bounds']) for unit_index in G.nodes if G.nodes[unit_index]['KS_bounds'][0][0] != 'None']
    if G.graph['aux_bounds'][0][0][0] != 'None': DS_bounds += [np.array(G.graph['aux_bounds']).reshape(np.array(G.graph['aux_bounds']).shape[0], np.array(G.graph['aux_bounds']).shape[2])]
    DS_bounds = np.vstack(DS_bounds)
    DS_bounds = pd.DataFrame(DS_bounds.T, columns=cfg.case_study.design_space_dimensions)
    return DS_bounds

def init_plot(cfg, G, pp= None, init=True, save=True):

    init_df = G.graph['initial_forward_pass']
    DS_bounds = get_ds_bounds(cfg, G)

    if init:
        pp = initializer_cp(init_df)
    else:
        assert pp is not None, "PairGrid object is None. Please provide a valid PairGrid object."
    
    pp.map_lower(sns.scatterplot, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5)
        
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

    if save: pp.savefig("initial_forward_pass.svg", dpi=DEFAULT_DPI)

    return pp

def decompose_call(cfg, G, path, init=True):
    pp = init_plot(cfg, G, init=init, save=False)
    pp = decomposition_plot(cfg, G, pp, save=True, path=path)
    return pp


def decomposition_plot(cfg, G, pp, save=True, path='decomposed_pair_grid_plot'):
    # load live sets for each subproblem from the graph 
    inside_samples_decom = [pd.DataFrame({col:G.nodes[node]['live_set_inner'][:,i] for i, col in enumerate(cfg.case_study.process_space_names[node])}) for node in G.nodes]
    print('cols', [{i: col for i, col in enumerate(cfg.case_study.process_space_names[node])} for node in G.nodes])

    print("inside_samples_decom", inside_samples_decom)
    # just keep those variables with Ui in the column name # TODO update this to also receive the live set probabilities 
    inside_samples_decom = [in_[[col for col in in_.columns if f"N{i+1}" in col]] for (i,in_) in enumerate(inside_samples_decom)]
    
    print("inside_samples_decom", inside_samples_decom)
    if cfg.reconstruction.plot_reconstruction == 'probability_map':
        for i, is_ in enumerate(inside_samples_decom):
            is_['probability'] = G.nodes[i]['live_set_inner_prob'] # TODO update this to also receive the live set probabilities

    # Get the indices of the lower triangle of the pair grid plot
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        for is_ in inside_samples_decom:
            if x_var in is_.columns and y_var in is_.columns:
                print(f"Plotting {x_var} vs {y_var}")
                sns.scatterplot(x=x_var, y=y_var, data=is_, edgecolor="k", c='r', alpha=0.8, ax=ax)
        
    if save: pp.savefig(path +'.svg', dpi=DEFAULT_DPI)

    return pp
    
def reconstruction_plot(cfg, G, reconstructed_df, save=True, path='reconstructed_pair_grid_plot'):

    pp = initializer_cp(reconstructed_df)
    pp = init_plot(cfg, G, pp, init=False, save=False)
    pp = decomposition_plot(cfg, G, pp, save =False)
    pp.map_lower(sns.scatterplot, data=reconstructed_df, edgecolor="k", c="b", linewidth=0.5)

    if save: pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return pp

def design_space_plot(cfg, G, joint_data_direct, path):

    # Pair-wise Scatter Plots
    pp = initializer_cp(joint_data_direct)
    DS_bounds = get_ds_bounds(cfg, G)
    
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

    pp.map_lower(sns.scatterplot, data=joint_data_direct, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return 


def design_space_plot_plus_polytope(cfg, G, pp, joint_data_direct, path, save=True):

    # Pair-wise Scatter Plots

    DS_bounds = get_ds_bounds(cfg, G)
    
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

    pp.map_lower(sns.scatterplot, data=joint_data_direct, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    if save: pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return pp

def polytope_plot(pp, polytope):

    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if (x_var,y_var) in list(polytope.keys()):
            ax.fill(polytope[(x_var,y_var)][0], polytope[(x_var,y_var)][1], alpha=0.5, color='red', edgecolor='black', linewidth=1.5)
    # Save the updated figure
    return pp

def polytope_plot_2(pp, polytope):
    from scipy.spatial import ConvexHull

    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in list(polytope.keys()) and y_var in list(polytope.keys()) and x_var[:2] == y_var[:2]:
            points = np.hstack([np.array(polytope[x_var]).reshape(-1,1), np.array(polytope[y_var]).reshape(-1,1)]).reshape(-1, 2)
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            # Unzip for plotting
            x_hull, y_hull = zip(*hull_vertices)
            ax.fill(x_hull, y_hull, alpha=0.5, color='red', edgecolor='black', linewidth=1.5)
    # Save the updated figure
    return pp

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
    return


def post_process_upper_solution(cfg, G, args):
    """
    Visualise the solution of the post-processing upper-level problem.
    :param cfg: Configuration object
    :param G: Graph object
    :param solution: Solution to be visualised
    :param path: Path to save the visualisation
    """
    solution, value_fn = args
    plotting_format()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Assuming solution is a DataFrame with columns: 'x', 'y', 'z'
    # where 'x' and 'y' are the 2D input variables and 'z' is the mapped 1D output
    xi, yi, zi = solution['x'], solution['y'], solution['z']
    # Plot the contour
    contour = ax.contourf(xi, yi, zi, vmin=0, vmax=np.max(zi), levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Point-wise error')

    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')

    plt.savefig("post_process_upper_solution.svg", dpi=DEFAULT_DPI)

    logging.info("Difference between predicted max point wise error and actual max point wise error: %s",
                 np.max(zi) - value_fn)
    
    return fig


def plot_contour(func, x_range, y_range, value_fn, path, num_points=200, levels=10):
    """
    Generates a contour plot for a given 2D function over a specified box domain.
    This version is designed for functions that can only be evaluated on a batch
    of points where the batch is the zeroth axis (e.g., func(x_coords, y_coords)
    where x_coords and y_coords are 1D arrays).

    Args:
        func (callable): The function to plot. It should accept two 1D arrays,
                        x and y, and return a single 1D array of results.
        x_range (tuple): A tuple (x_min, x_max) defining the range for the x-axis.
        y_range (tuple): A tuple (y_min, y_max) defining the range for the y-axis.
        num_points (int, optional): The number of points to use for the grid along
                                    each axis. A higher number results in a smoother
                                    plot. Defaults to 200.
        levels (int, optional): The number of contour lines or filled regions to display.
                                Defaults to 10.
    """
    # Create a grid of points for the x and y axes
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    
    # Use numpy.meshgrid to create the 2D grid from the 1D arrays
    X, Y = np.meshgrid(x, y)

    # Flatten the X and Y grids into 1D arrays for the batch evaluation
    # This creates a "batch" of all coordinate pairs to be evaluated
    x_coords_batch = X.ravel()
    y_coords_batch = Y.ravel()

    # Evaluate the input function on the batch of points
    # The function is expected to return a 1D array of results
    z_values_batch = func(x_coords_batch, y_coords_batch)

    # Reshape the 1D result back into a 2D array with the original grid shape
    Z = z_values_batch.reshape(X.shape)

    # Create the plot figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # --- Generate the contour plot ---
    # The 'contourf' function creates filled contour regions.
    # The 'cmap' parameter sets the colormap.
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    
    # The 'contour' function creates the contour lines on top of the filled regions.
    # We use 'colors='k'' to make the lines black.
    ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.5)

    # --- Customize the plot ---
    # Add a color bar to show the mapping from color to Z-value
    fig.colorbar(cf, ax=ax, vmin=np.min(Z), vmax=np.max(Z), label='Point-wise error')

    # Add titles and labels
    #ax.set_title(f'Contour Plot of {func.__name__}', fontsize=16)
    ax.set_xlabel(f'r$x_1$', fontsize=12)
    ax.set_ylabel(f'r$x_2$', fontsize=12)
    ax.set_aspect('equal', adjustable='box') # Ensure the plot is not stretched

    logging.info("Difference between predicted max point wise error and actual max point wise error: %s",
                 np.max(Z) - value_fn)

    # Display the plot
    plt.savefig(os.path.join(path, "post_process_upper_solution.svg"), dpi=DEFAULT_DPI)