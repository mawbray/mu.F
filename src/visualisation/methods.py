import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotting_format():
    font = {"family": "serif", "weight": "bold", "size": 20}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rc("axes", titlesize=0)  # fontsize of the axes title
    plt.rc("axes", labelsize=15)  # fontsize of the x and y label
    plt.rc("axes", linewidth=3)
    plt.rc("axes", labelpad=20)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    return


def method_plot(save_folder, cfg, G, df):
    inside_samples_decom = [pd.read_excel(os.path.join(save_folder,f'inside_samples_{i+1}.xlsx')) for i in range(3)]
    outside_samples_decom = [pd.read_excel(os.path.join(save_folder,f'outside_samples_{i+1}.xlsx')) for i in range(3)]

    # just keep those variables with U2 in the column name
    inside_samples_decom = [in_[[col for col in in_.columns if f"U{i+1}" in col]] for (i,in_) in enumerate(inside_samples_decom)]
    outside_samples_decom = [in_[[col for col in in_.columns if f"U{i+1}" in col]] for (i,in_) in enumerate(outside_samples_decom)]

    ## plot a scatter of the DS using those intialised samples in the forward pass
    init_df = G.graph['initial_forward_pass']
    DS_bounds = [np.array(G.nodes[unit_index]['KS_bounds']) for unit_index in G.nodes]
    DS_bounds = np.vstack(DS_bounds)
    DS_bounds = pd.DataFrame(DS_bounds.T, columns=cfg.design_space_dimensions)
    init_df.rename(columns={"U3: P.C. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: M.C. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)
    DS_bounds.rename(columns={"U3: P.C. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: M.C. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)
    design_space_names = cfg.design_space_dimensions
    process_space_names = cfg.process_space_names
    subsets_names = []
    for ps in process_space_names:
        x = [p for p in ps if p in design_space_names]
        subsets_names.append(x)

    #inside_samples_restrict = [in_[cols] for in_, cols in zip(inside_samples_decom, subsets_names)]
    #outside_samples_restrict = [in_[cols] for in_, cols in zip(outside_samples_decom, subsets_names)]
    print(df)
    df.rename(columns={"U3: P.C. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: M.C. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)
    print(df)
    print(inside_samples_decom[-1])
    print(outside_samples_decom[-1])
    inside_samples_decom[-1].rename(columns={"U3: Pre-comp. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: Main-comp. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)
    outside_samples_decom[-1].rename(columns={"U3: Pre-comp. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: Main-comp. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)

    print(inside_samples_decom[-1])
    print(outside_samples_decom[-1])

    plotting_format()
    fig = plt.figure(figsize=(35, 35))
    cols = init_df.columns
    pp = sns.PairGrid(
        init_df[cols],
        aspect=1.4
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)

    pp.map_lower(sns.scatterplot, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5)
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')


    pp.savefig("initial_forward_pass.svg", dpi=300)

    # Pair-wise Scatter Plots
    fig = plt.figure(figsize=(35, 35))
    cols = df.columns
    pp = sns.PairGrid(
        init_df[cols],
        aspect=1.4,
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)

    pp.map_lower(sns.scatterplot, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5)
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            print(x_var, y_var)
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')


    # Get the indices of the lower triangle of the pair grid plot
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)
    print(outside_samples_decom)
    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        #for os_ in outside_samples_decom:
        #    print(os_.columns)
        #    if x_var in os_.columns and y_var in os_.columns:
        #        sns.scatterplot(x=x_var, y=y_var, data=os_, edgecolor="k", c='k', alpha=0.1, ax=ax) 
        for is_ in inside_samples_decom:
            print(is_.columns)
            if x_var in is_.columns and y_var in is_.columns:
                sns.scatterplot(x=x_var, y=y_var, data=is_, edgecolor="k", c='r', alpha=0.8, ax=ax)
        

    #pp.map_lower(sns.scatterplot, data=df, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    pp.savefig("decomposed_pair_grid_plot.svg", dpi=300)

    ### plot final boss -- full reconstruction


    # Pair-wise Scatter Plots
    fig = plt.figure(figsize=(35, 35))
    cols = df.columns
    pp = sns.PairGrid(
        df[cols],
        aspect=1.4,
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)

    pp.map_lower(sns.scatterplot, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5)
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')



    # Get the indices of the lower triangle of the pair grid plot
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)
    print(outside_samples_decom)
    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        #for os_ in outside_samples_decom:
        #    print(os_.columns)
        #    if x_var in os_.columns and y_var in os_.columns:
        #        sns.scatterplot(x=x_var, y=y_var, data=os_, edgecolor="k", c='k', alpha=0.1, ax=ax) 
        for is_ in inside_samples_decom:
            print(is_.columns)
            if x_var in is_.columns and y_var in is_.columns:
                sns.scatterplot(x=x_var, y=y_var, data=is_, edgecolor="k", c='r', alpha=0.8, ax=ax)
        

    pp.map_lower(sns.scatterplot, data=df, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    pp.savefig("updated_pair_grid_plot.svg", dpi=300)

    date_time = 'outputs/2024-04-11/16-25-35'
    save_file = os.path.join(cwd, os.path.join(date_time,'joint_inside_samples.xlsx'))
    joint_data_direct = pd.read_excel(save_file, index_col=0)
    joint_data_direct.rename(columns={"U3: P.C. Pressure (MPa)": "U3: P.C.P. (MPa)", "U3: M.C. Pressure (MPa)": "U3: M.C.P. (MPa)"}, inplace=True)

    # Pair-wise Scatter Plots
    fig = plt.figure(figsize=(35, 35))
    cols = joint_data_direct.columns
    pp = sns.PairGrid(
        joint_data_direct[cols],
        aspect=1.4,
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)

    indices = zip(*np.tril_indices_from(pp.axes, -1))
    #print([(i,j) for i,j in indices])
    print(pp.x_vars)
    print(pp.y_vars)

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
    pp.savefig("direct_pair_grid_plot.svg", dpi=300)

    return



def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
    return


