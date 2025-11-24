
def estimator_regressor_data_function(candidates, constraints, desired_node_index):
    """
    A function to process regressor data for convex estimator
    :param constraints: The constraint values
    :param candidates: The candidate points
    :return: The processed edge data
    """
    inputs = candidates[:,:-1]
    outputs = (candidates[:,-1].reshape(-1,1) - constraints[desired_node_index].reshape(-1,1))**2
    return inputs, outputs



post_process_regressor_data_function = {"tablet_press": lambda x, y, z: (x, y),
"serial_mechanism_batch": lambda x, y, z: (x, y),
"convex_estimator": estimator_regressor_data_function,
"convex_underestimator": estimator_regressor_data_function,
"estimator": estimator_regressor_data_function,
"affine_study": lambda x, y, z: (x, y)
}