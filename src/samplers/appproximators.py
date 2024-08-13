import jax.numpy as jnp 

def calculate_box_outer_approximation(data, config, ndim=3):
    """
    Calculate the box outer approximation of the given data.

    Parameters:
    data (jnp.array): The input data.
    config (object): The configuration object with a 'samplers.vol_scale' attribute.

    Returns:
    list: The minimum and maximum values of the box outer approximation.
    """

    if ndim == 3: 
        # reshape data to account for the uncertainty parameters
        if data.ndim < 3:
            data = jnp.expand_dims(data, axis=-1)
        data = jnp.vstack([data[:,i,:].reshape(data.shape[0], data.shape[2]) for i in range(data.shape[1])])


    # Calculate the range of the input data
    data_range = jnp.max(data, axis=0) - jnp.min(data, axis=0)

    # Calculate the increment/decrement value for the box outer approximation
    delta = config.samplers.vol_scale / 2 * data_range

    # Calculate the minimum and maximum values of the box outer approximation
    min_value = jnp.min(data - delta, axis=0)
    max_value = jnp.max(data + delta, axis=0)

    return [min_value.reshape(1,-1), max_value.reshape(1,-1)]