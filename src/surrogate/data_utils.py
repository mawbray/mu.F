import jax.numpy as jnp
import pandas as pd
from omegaconf import DictConfig



def evaluate_classifier(classifier, data_points, cfg, index):

    predictions = classifier.predict(data_points)
    mapping = {'Prediction': predictions} | {name: data_points[:,i]*cfg.scale[index][i] for i,name in enumerate(cfg.process_space_names[index])}
    df = pd.DataFrame(mapping)

    return df


def binary_classifier_data_preparation(
    graph: dict,
    unit_index: int,
    cfg: DictConfig = None,
):
    # access historical support and function values

    data = graph.nodes[unit_index]["classifier_training"]
    support = data.X
    labels = data.y
    if cfg.model == 'binary classifier': # TODO update live_set_surrogate to model
        if cfg.notion_of_feasibility == 'positive':
            select_cond = jnp.min(labels, axis=1)  >= 0 # 
        else:
            select_cond = jnp.max(labels, axis=1)  <= 0  
        labels = jnp.where(select_cond, -1, 1) # binary classifier (feasible label is always negative because we are always minimizing in problem coupling, just depends on which data we label)
    else:
        raise NotImplementedError('Only binary classifier is supported for now')

    support, labels = return_subsample_of_data(support, labels, cfg.coupling_constraint.subsample_size)

    return support, labels

def return_subsample_of_data(data, labels, subsample_size):
    if data.shape[0] > subsample_size:
        select_cond = labels == 1 # 
        data_pos = data[select_cond]
        select_cond = labels == -1 #
        data_neg = data[select_cond]
        assert subsample_size > data_neg.shape[0], f"Negative data size {data_neg.shape[0]} is larger than subsample size {subsample_size}"

        data_new = jnp.vstack([data_neg, data_pos[:subsample_size-data_neg.shape[0],:]])
        labels_new = jnp.vstack([jnp.ones((data_neg.shape[0],1)), jnp.ones((subsample_size-data_neg.shape[0],1))*-1 ])
        return data_new, labels_new
    else:
        return data, labels