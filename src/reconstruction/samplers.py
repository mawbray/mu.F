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