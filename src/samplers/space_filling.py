
import numpy as np
from scipy.stats.qmc import Sobol

class sobol_sampler:
    def __init__(self):
        pass

    def sample_design_space(self, n_design_args, bounds, n):
        return sobol_sample_design_space_nd(n_design_args, bounds, n)

def sobol_sample_design_space_nd(n_design_args, bounds, n):

    # better to sample using whole dimensionality of design space to ensure coverage
    sobol_values = Sobol(n_design_args).random(n)

    lower_bound = np.array(bounds[0])
    upper_bound = np.array(bounds[1])
    # Scale the values to lie within the specified range
    design_args = lower_bound + (upper_bound - lower_bound) * sobol_values[:, :]

    return design_args




def measure_live_set_volume(G, design_space_dimensions):
    """
    Measure the volume of the live set in the design space.
    """
    # Get the design space dimensions
    design_space_dimensions = design_space_dimensions

    # Get the live set
    live_set = np.array([G.nodes[node]['live_set_inner'] for node in G.nodes()])

    # Get the number of points in the live set
    n_points = live_set.shape[0]

    # Get the number of dimensions in the design space
    n_dimensions = len(design_space_dimensions)

    # Get the volume of the live set
    volume = np.prod(np.max(live_set, axis=0) - np.min(live_set, axis=0))

    return volume