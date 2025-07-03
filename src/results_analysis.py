# this is a file to analyse results 
from unit_evaluators.constructor import network_simulator
from reconstruction.constructor import reconstruction
from constraints.constructor import constraint_evaluator
from visualisation.methods import plotting_format
from cs_assembly import case_study_allocation, solver_constructor, unit_params_fn
from constraints.functions import CS_holder
from graph.methods import CS_edge_holder, vmap_CS_edge_holder
from graph.graph_assembly import graph_constructor

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histogram_unit(cons_g, unit, path):
    # Define the label for the x-axis.
    x_axis_label = "Constraint value for unit {}".format(unit)

    # Define the label for the y-axis.
    y_axis_label = "Frequency"

    # Define the color for the histogram bars.
    bar_color = 'skyblue'

    # Define the color for the edges of the histogram bars.
    edge_color = 'black'

    # Define the line style for the grid.
    grid_linestyle = '--'

    # Define the alpha (transparency) for the grid.
    grid_alpha = 0.7

    # --- Plotting ---
    def create_histogram(data, num_bins, plot_title, x_axis_label, y_axis_label, bar_color, edge_color, grid_linestyle, grid_alpha, path):
        """
        Creates and displays a histogram from the given data.

        Args:
            data (np.array): The array of numerical data to plot.
            num_bins (int): The number of bins for the histogram.
            plot_title (str): The title of the plot.
            x_axis_label (str): The label for the x-axis.
            y_axis_label (str): The label for the y-axis.
            bar_color (str): Color of the histogram bars.
            edge_color (str): Color of the histogram bar edges.
            grid_linestyle (str): Line style for the grid.
            grid_alpha (float): Transparency of the grid.
        """
        plt.figure(figsize=(10, 6)) # Set the figure size for better readability

        # Create the histogram
        # `hist` returns:
        # - `n`: The values of the histogram bins (frequency).
        # - `bins`: The edges of the bins.
        # - `patches`: The individual bars of the histogram.
        plt.hist(data, bins=num_bins, color=bar_color, edgecolor=edge_color)

        # Add plot title and labels
        plt.title(plot_title, fontsize=16)
        plt.xlabel(x_axis_label, fontsize=12)
        plt.ylabel(y_axis_label, fontsize=12)

        # Add a vertical line at x=0 to clearly show the boundary.
        # This line emphasizes that all data points are to the left of or at zero.
        plt.axvline(x=0, color='red', linestyle=':', linewidth=2, label='Zero Line')

        # Add a legend to explain the red line.
        plt.legend()

        # Add a grid for better readability
        plt.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

        # Set x-axis limits to ensure 0 is visible and the plot looks clean.
        # We set the maximum x-limit to a small positive value (e.g., 0.5)
        # to ensure the "Zero Line" is clearly visible and there's no data
        # extending beyond it. The minimum is set slightly below the min data value.
        plt.xlim(min(data) - 0.5, 0.5)

        # Display the plot
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(path + f'/histogram_unit_{unit}.svg', dpi=300) # Save the plot as a PNG file

    create_histogram(
        data=cons_g,
        num_bins=100,  # Number of bins for the histogram
        plot_title=f"Histogram of Constraint Values for Unit {unit}",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        bar_color=bar_color,
        edge_color=edge_color,
        grid_linestyle=grid_linestyle,
        grid_alpha=grid_alpha,
        path = path
    )
    
def graph_reconstruction(cfg, graph):
    """
    Reconstruct the graph with the given configuration and graph.
    :param cfg: The configuration object
    :param graph: The graph object
    :return: The reconstructed graph
    """
    # Create a sample constraint dictionary
    constraint_dictionary = CS_holder[cfg.case_study.case_study]

    # create edge functions
    if cfg.case_study.vmap_evaluations:
        dict_of_edge_fn = vmap_CS_edge_holder[cfg.case_study.case_study]
    else:
        dict_of_edge_fn = CS_edge_holder[cfg.case_study.case_study]

    # construct dummy dataframe for initial forward pass
    init_df_samples = pd.DataFrame({col: np.zeros((2,)) for i,col in enumerate(cfg.case_study.design_space_dimensions)})


    # Create a case study allocation object
    G = case_study_allocation(graph, cfg, dict_of_edge_fn, constraint_dictionary, solvers=solver_constructor(cfg, graph), unit_params_fn=unit_params_fn(cfg, graph), initial_forward_pass=init_df_samples)
    graph = G.get_graph()
    return graph


def evaluate(cfg, graph, n_live, path):

    G= graph_constructor(cfg, cfg.case_study.adjacency_matrix)
    G.load_graph(graph)
    graph = graph_reconstruction(cfg, G)
    network_model = network_simulator(cfg, graph, constraint_evaluator)
    reconstructor = reconstruction(cfg, graph, network_model, 10)
    OmegaConf.update(cfg, key='samplers.ns.final_sample_live', value=n_live)
    feasible_candidates, _ = reconstructor.run()
    print('Acceptance ratio:', reconstructor.ls_holder.acceptanceratio)

    uncertain = reconstructor.get_uncertain_params()
    cons_g = reconstructor.evaluate_joint_model(feasible_candidates, uncertain_params=uncertain)

    for unit, g in cons_g.items():
        plot_histogram_unit(-g, unit, path=path)


    return 

if __name__ == '__main__':

    import os 
    import pickle as pkl
    import sys 

    root    = 'paper_results/decentralised/19-42-10/0'
    config_path = os.path.join(root, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    graph_stem = 'graph_backward-forward_iterate_10.pickle'

    with open(os.path.join(root, graph_stem), 'rb') as file:
        sys.path.append(os.getcwd())
        graph = pkl.load(file)

    evaluate(cfg, graph, n_live=10000, path = root)
