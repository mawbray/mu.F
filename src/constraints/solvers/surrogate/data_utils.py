import jax.numpy as jnp
import jax
import pandas as pd
from omegaconf import DictConfig
from dataclasses import dataclass




@dataclass 
class standardisation_metrics:
    mean: jnp.ndarray
    std: jnp.ndarray




def evaluate_classifier(classifier, data_points, cfg, index):

    predictions = classifier.predict(data_points)
    mapping = {'Prediction': predictions} | {name: data_points[:,i]*cfg.scale[index][i] for i,name in enumerate(cfg.process_space_names[index])}
    df = pd.DataFrame(mapping)

    return df

def forward_evaluation_data_preparation(graph: dict, unit_index, cfg: DictConfig = None, successor_node: int = None):
    # access historical support and function values
    input_data = graph.edges[unit_index, successor_node]["surrogate_training"]

    return input_data

def regression_node_data_preparation(graph: dict, unit_index: int, cfg: DictConfig = None):
    # access historical support and function values
    data = graph.nodes[unit_index]["probability_map_training"]

    return data


def binary_classifier_data_preparation(
    graph: dict,
    unit_index: int,
    cfg: DictConfig = None,
):
    # access historical support and function values

    data = graph.nodes[unit_index]["classifier_training"]
    support = data.X
    labels = data.y
    
    if cfg.formulation == 'deterministic':
        if cfg.samplers.notion_of_feasibility == 'positive':
            select_cond = jnp.min(labels, axis=1)  >= 0 # 
        else:
            select_cond = jnp.max(labels, axis=1)  <= 0  
    elif cfg.formulation == 'probabilistic':
        select_cond = labels >= cfg.samplers.unit_wise_target_reliability[unit_index]
    else: 
        raise ValueError(f"Formulation {cfg.formulation} not recognised. Please use 'probabilistic' or 'deterministic'.")

    labels = jnp.where(select_cond, -1, 1) # binary classifier (feasible label is always negative because we are always minimizing in problem coupling, just depends on which data we label)

    # Data augmentation to equalize the number of negative and positive classes
    num_pos = jnp.sum(labels == 1)
    num_neg = jnp.sum(labels == -1)
    Key = jax.random.PRNGKey(0)
    # Add 1% Gaussian noise to the datapoints in the support
    
    if num_pos > num_neg:
        # Randomly select negative samples to match the number of positive samples
        neg_indices = jnp.where(labels == -1)[0]
        selected_indices = jax.random.choice(Key, neg_indices, shape=(num_pos - num_neg,))
        support = jnp.concatenate([support.squeeze(), support[selected_indices].squeeze()], axis=0)
        labels = jnp.concatenate([labels, labels[selected_indices]], axis=0)
    elif num_neg > num_pos:
        # Randomly select positive samples to match the number of negative samples
        pos_indices = jnp.where(labels == 1)[0]
        selected_indices = jax.random.choice(Key, pos_indices, shape=(num_neg - num_pos,))
        support = jnp.concatenate([support.squeeze(), support[selected_indices].squeeze()], axis=0)
        labels = jnp.concatenate([labels, labels[selected_indices]], axis=0)


    # TODO think about this
    #support, labels = return_subsample_of_data(support, labels, cfg.surrogate.subsample_size)
    

    return support, labels



def return_subsample_of_data(data, labels, subsample_size):
    if data.shape[0] > subsample_size:
        select_cond = labels == 1 # 
        data_pos = data[select_cond.squeeze(),:]
        select_cond = labels == -1 #
        data_neg = data[select_cond.squeeze(),:]
        assert subsample_size > data_neg.shape[0], f"Negative data size {data_neg.shape[0]} is larger than subsample size {subsample_size}"

        data_new = jnp.vstack([data_neg, data_pos[:subsample_size-data_neg.shape[0],:]])
        labels_new = jnp.vstack([-jnp.ones((data_neg.shape[0],1)), jnp.ones((subsample_size-data_neg.shape[0],1))*1 ])
        return data_new, labels_new
    else:
        return data, labels