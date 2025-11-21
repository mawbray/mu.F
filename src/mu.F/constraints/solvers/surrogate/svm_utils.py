import pandas as pd

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

# hydra imports
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

import jax.numpy as jnp
from jax import jit

from constraints.solvers.surrogate.data_utils import binary_classifier_data_preparation, standardisation_metrics

import logging

@jit
def all_feasible(x):
    return jnp.linalg.norm(x*jnp.array([1e-4]), ord='fro') + jnp.array([-5])

@jit
def no_feasible(x):
    return jnp.linalg.norm(x*jnp.array([1e-4]), ord='fro') + jnp.array([5])

def get_serialised_model_data(model):

    return model.get_serialised_model_data()

def train(cfg, dataset, num_folds, unit_index, iterate):
    """
    Construct the classifier for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'classifier' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """
    data_points, labels = dataset
    # construct a classifier based on the data available # NOTE simple classifier for now
    s_vectors, classifier, p_dict, labels = compute_best_svm_classifier(
        data_points, labels, unit_index=unit_index, iterate=iterate, cfg=cfg, num_folds=num_folds
    )

    if classifier is not None:
        _, args, model_data = convert_svm_to_jax(classifier.best_estimator_)
        return classifier, args, model_data
    else:
        if jnp.all(labels == -1):
            classifier = all_feasible
            model_data = get_model_data(
            jnp.zeros_like(data_points[0,:]), 
            jnp.ones_like(data_points[0,:]), 
            jnp.zeros((1,)),
            jnp.zeros_like(data_points[0,:]), 
            jnp.array([-5.]), 
            1.0
        )
        elif jnp.all(labels == 1):
            classifier = no_feasible
            model_data = get_model_data(
            jnp.zeros_like(data_points[0,:]), 
            jnp.ones_like(data_points[0,:]), 
            jnp.zeros((1,)),
            jnp.zeros_like(data_points[0,:]), 
            jnp.array([-5.]), 
            1.0
        )
        args = (classifier, classifier, None)
        

        return classifier, args, model_data

def compute_best_svm_classifier(
    data_points, labels, unit_index, cfg, iterate, num_folds
): 
    
    # build classification model
    x_r = sum(labels) / labels.shape[0]
    clf = Pipeline([("scaler", StandardScaler()), ("svc", svm.SVC())])
    parameters = [
        {
            "svc__kernel": cfg.surrogate.classifier_args.svm.kernel,
            "svc__C": cfg.surrogate.classifier_args.svm.C,
            "svc__gamma": cfg.surrogate.classifier_args.svm.gamma,
            "svc__probability": cfg.surrogate.classifier_args.svm.probability,
        }
    ]
    # grid search over parameters
    model = GridSearchCV(clf, parameters, scoring="accuracy", cv=num_folds)
    try:
        if data_points.ndim > 2: data_points = data_points.squeeze()
        model.fit(data_points, labels.squeeze())
    except: 
        logging.info("error in fitting classification model")

        return None, None, None, labels
    # get support vectors
    support_vectors = model.best_estimator_["svc"].support_vectors_
    # get classifier performance
    accuracy = model.score(data_points, labels)
    # get classifier false positive rates
    tn, fp, fn, tp = confusion_matrix(model.predict(data_points), labels).ravel()
    # metrics compression
    training_performance = {"acc": accuracy, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

    logging.info(f"training_performance: {training_performance}")


    df = pd.DataFrame(model.cv_results_)
    df = df.sort_values(by=["rank_test_score"])
    df.to_excel(f"svm_cv_results_{unit_index}_{iterate}.xlsx")

    return support_vectors, model, training_performance, labels
    
  
@jit
def rbf_kernel(x, y, epsilon=1e-3):
    # Compute the RBF kernel for multivariate case, this implementation is reasonable and avoids constructing large matrices.
    diff = x - y
    squared_diff = jnp.linalg.norm(diff, axis=1)**2
    return jnp.exp(-epsilon * squared_diff)

def get_x_scalar_from_pipeline(pipeline):
    return pipeline[0]

def convert_svm_to_jax(pipeline):

    # Extract the support vectors and coefficients from the SVM model
    svm_model = pipeline[1]
    support_vectors = svm_model.support_vectors_
    y_data = pipeline.predict(support_vectors).reshape(-1,1)
    coefficients = svm_model.dual_coef_[0].reshape(1,-1) # only works for binary classifier length of dual_coef_ is n_support_vectors
    intercept = svm_model.intercept_
    kernel_param = svm_model.gamma

    # access standardisation parameters
    x_scalar = pipeline[0]
    x_mean = jnp.array(x_scalar.mean_)
    x_std = jnp.array(x_scalar.scale_)

    # Define the JAX SVM model
    @jit
    def svm_unstandardised(x):
        # Compute the decision function
        x_ = (x.reshape(-1,) - x_mean.reshape(-1,)) / x_std.reshape(-1,)
        x_ = x_.reshape(1, -1)
        decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x_,epsilon=kernel_param)) + intercept
        # Apply the sign function to get the predicted class
        return decision.squeeze()

    @jit
    def svm_standardised(x):
        # defining multivariate -> univariate fn (no batching support )
        x = x.reshape(1, -1)
        decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x,epsilon=kernel_param)) + intercept
        return decision.squeeze()
    
    model_data = get_model_data(x_mean, x_std, support_vectors, coefficients, intercept, kernel_param)
    
    return svm_model, (svm_standardised, svm_unstandardised, standardisation_metrics(mean=x_mean, std=x_std)), model_data

def get_model_data(x_mean, x_std, support_vectors, coefficients, intercept, kernel_param):
    
    model_data = {'standardisation_metrics_input': standardisation_metrics(mean=x_mean, std=x_std), 
                  'serialized_params': {'support_vectors': support_vectors,
                                        'coefficients': coefficients,
                                        'intercept': intercept,
                                        'kernel_param': kernel_param}}
    return model_data

def build_svm(cfg, model_data):

    x_standardisation = model_data['standardisation_metrics_input']
    x_mean = x_standardisation.mean
    x_std = x_standardisation.std


    # Define the SVM model
    support_vectors = model_data['serialized_params']['support_vectors']
    coefficients = model_data['serialized_params']['coefficients']
    intercept = model_data['serialized_params']['intercept']
    kernel_param = model_data['serialized_params']['kernel_param']

    if cfg['solvers']['standardised']:
        @jit
        def svm_standardised(x):
            # defining multivariate -> univariate fn (no batching support )
            x = x.reshape(1, -1)
            decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x,epsilon=kernel_param)) + intercept
            return decision.reshape(-1,1)
        return svm_standardised
        
    else:
        @jit
        def svm_unstandardised(x):
            # Compute the decision function
            x_ = (x.reshape(-1,) - x_mean.reshape(-1,)) / x_std.reshape(-1,)
            x_ = x_.reshape(1, -1)
            decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x_,epsilon=kernel_param)) + intercept
            # Apply the sign function to get the predicted class
            return decision.reshape(-1,1)
        return svm_unstandardised
        




