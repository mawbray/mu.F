import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotting_format():
    font = {"family": "serif", "weight": "bold", "size": 20}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rc("axes", labelsize=15)  # fontsize of the x and y label
    plt.rc("axes", linewidth=3)
    plt.rc("axes", labelpad=20)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

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
    if G.graph['aux_bounds'][0][0] != 'None': DS_bounds += [np.array(G.graph['aux_bounds']).reshape(np.array(G.graph['aux_bounds']).shape[0], np.array(G.graph['aux_bounds']).shape[2])]
    DS_bounds = np.vstack(DS_bounds)
    DS_bounds = pd.DataFrame(DS_bounds.T, columns=cfg.case_study.design_space_dimensions)
    return DS_bounds

def init_plot(cfg, G, pp= None, init=True, save=True):
    init_df = G.graph['initial_forward_pass']
    DS_bounds = get_ds_bounds(cfg, G)

    if init:
        pp = initializer_cp(init_df)

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

    if save: pp.savefig("initial_forward_pass.svg", dpi=300)

    return pp

def decompose_call(cfg, G, path):
    pp = init_plot(cfg, G, init=True, save=False)
    pp = decomposition_plot(cfg, G, pp, save=True, path=path)
    return pp


def decomposition_plot(cfg, G, pp, save=True, path='decomposed_pair_grid_plot'):
    # load live sets for each subproblem from the graph 
    inside_samples_decom = [pd.DataFrame({col:G.nodes[node]['live_set_inner'][:,i] for i, col in enumerate(cfg.case_study.process_space_names[node])}) for node in G.nodes]

    # just keep those variables with Ui in the column name # TODO update this to also receive the live set probabilities 
    inside_samples_decom = [in_[[col for col in in_.columns if f"U{i+1}" in col]] for (i,in_) in enumerate(inside_samples_decom)]
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
                sns.scatterplot(x=x_var, y=y_var, data=is_, edgecolor="k", c='r', alpha=0.8, ax=ax)
        
    if save: pp.savefig(path +'.svg', dpi=300)

    return pp
    
def reconstruction_plot(cfg, G, reconstructed_df, save=True, path='reconstructed_pair_grid_plot'):

    pp = initializer_cp(reconstructed_df)
    pp = init_plot(cfg, G, pp, init=False, save=False)
    pp = decomposition_plot(cfg, G, pp, save =False)
    pp.map_lower(sns.scatterplot, data=reconstructed_df, edgecolor="k", c="b", linewidth=0.5)

    if save: pp.savefig(path + ".svg", dpi=300)

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
    pp.savefig(path + ".svg", dpi=300)

    return 


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
    return


