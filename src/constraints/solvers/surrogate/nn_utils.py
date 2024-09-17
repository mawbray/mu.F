from typing import Dict, Any
from itertools import product
from abc import ABC
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
from jax import value_and_grad, jit, vmap


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from flax import linen as nn
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot
from flax.training.early_stopping import EarlyStopping
from flax.experimental.nnx import softmax
from flax.serialization import to_bytes, from_bytes
from flax import jax_utils
import optax
from sklearn.metrics import confusion_matrix
from datetime import datetime

import unittest
from unittest.mock import Mock
import logging
from functools import partial
from omegaconf import DictConfig
from functools import partial

from constraints.solvers.surrogate.data_utils import standardisation_metrics



class Dataset(ABC):
    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else jnp.expand_dims(X, axis=-1)
        self.y = y if self.output_rank >=2 else jnp.expand_dims(y,axis=-1)


# --- neural network regressor --- #

def identify_neural_network(hidden_units, output_units, activation_functions) -> nn.Module:
    return NeuralNetworkEstimator(hidden_units=hidden_units, output_units=output_units, activation_functions=activation_functions)

def serialise_model(params, model, x_scalar, y_scalar, model_type, model_data):
    model_data['hidden_units'] = model.hidden_units
    model_data['output_units'] = model.output_units
    model_data['activation_function'] = model.activation_functions
    model_data['serialized_params'] = to_bytes(params)
    model_data['standardisation_metrics_input'] = standardisation_metrics(x_scalar.mean_, x_scalar.scale_)
    if model_type == 'regressor':
        model_data['standardisation_metrics_output'] = standardisation_metrics(y_scalar.mean_, y_scalar.scale_)

    #assess model serialisation 
    # Function to compare original and deserialized parameters
    def compare_params(original, deserialized):
        for k, v in original.items():
            if isinstance(v, dict):
                compare_params(v, deserialized[k])
            else:
                assert jnp.allclose(v, deserialized[k]), f"Mismatch found in parameter {k}"

    # Compare original and deserialized parameters
    try:
        compare_params(params, from_bytes(params, model_data['serialized_params']))
        print("Serialization and deserialization successful. Parameters match!")
    except AssertionError as e:
        print(f"Error: {e}")
    return model_data


def hyperparameter_selection(cfg: DictConfig, D, num_folds: int, model_type, rng_key: random.PRNGKey=jax.random.PRNGKey(0)): 
    # Define the hyperparameters to search over
    surrogate_cfg = cfg.surrogate.surrogate_forward.ann
    hidden_sizes = surrogate_cfg.hidden_size_options
    afs = surrogate_cfg.activation_functions

    # Initialize the best hyperparameters and the best average validation loss
    best_hyperparams = {}
    best_avg_loss = float('inf')

    x_scalar = StandardScaler().fit(D.X)
    if model_type == 'regressor':
        y_scalar = StandardScaler().fit(D.y)
        standard_D = Dataset(x_scalar.transform(D.X), y_scalar.transform(D.y))
    elif model_type == 'classifier':
        y_scalar = jnp.astype((D.y + 1)/2, jnp.int32)
        standard_D = Dataset(x_scalar.transform(D.X), y_scalar)

    

    # Perform hyperparameter selection using cross-validation
    for hidden_size, af in product(hidden_sizes, afs):
        # Set the current hyperparameters
        # Train the model using the current hyperparameters
        if model_type == 'regressor':
            model = identify_neural_network(hidden_size, standard_D.y.shape[1], af)
        elif model_type == 'classifier':
            model = identify_neural_network(hidden_size, 2, af)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        avg_loss = train_nn_surrogate_model(surrogate_cfg, standard_D, model, num_folds, rng_key, model_type=model_type)

        # Check if the current hyperparameters are the best so far
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_hyperparams = {
                'hidden_size': hidden_size,
                'activation_function': af
            }

    # Train the model with the best hyperparameters using all the data
    if model_type == 'regressor':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], standard_D.y.shape[1], best_hyperparams['activation_function'])
        best_hyperparams['output_units'] = standard_D.y.shape[1]
    elif model_type == 'classifier':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], 2, best_hyperparams['activation_function'])
        best_hyperparams['output_units'] = 2
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    best_params, _, _ = train(surrogate_cfg, best_model, standard_D, standard_D, model_type=model_type) # train on standardised data

    opt_model = partial(best_model.apply, best_params)
    x_mean = jnp.array(x_scalar.mean_)
    x_std = jnp.array(x_scalar.scale_)
    
    logging.info(f'--- {model_type} ---')
    logging.info(f"Best hyperparameters: {best_hyperparams}")
    logging.info(f"Best average validation loss: {best_avg_loss}")

    serialised_model = serialise_model(best_params, best_model, x_scalar, y_scalar, model_type, {})

    @jit
    def standardise(x):
        return (x - x_mean) / x_std

    if model_type == 'regressor':

        y_mean = jnp.array(y_scalar.mean_)
        y_std = jnp.array(y_scalar.scale_)

        @jit
        def project(y):
            return y * y_std + y_mean

        @jit
        def query_unstandardised_model(x):
            if x.ndim <2 : x = x.reshape(1,-1)
            return project(opt_model(standardise(x))) # pipeline model

        @jit
        def query_standardised_model(x):
            if x.ndim <2: x = x.reshape(1,-1)
            return opt_model(x)

        return opt_model, (query_standardised_model, query_unstandardised_model, standardisation_metrics(x_mean, x_std), standardisation_metrics(y_mean, y_std)), serialised_model
    
    elif model_type == 'classifier':

        def mapp_(y):
            if y.ndim <2 : y = y.reshape(1,-1)
            return jnp.array([0.5]) - softmax(y, axis=-1)[0,0]    # if this quantity is less than or equal to zero, then the sample predicts feasibility.

        @jit
        def query_unstandardised_classifier(x):
            if x.ndim <2 : x = x.reshape(1,-1)
            return mapp_(opt_model(standardise(x))) # pipeline model
        
        @jit
        def query_standardised_classifier(x):
            if x.ndim <2: x = x.reshape(1,-1)
            if x.shape[0]>= x.shape[1]: x= x.T
            return mapp_(opt_model(x))
        
        return opt_model, (query_standardised_classifier, query_unstandardised_classifier, standardisation_metrics(x_mean, x_std)), serialised_model #, standardisation_metrics(jnp.zeros((1,2)), jnp.ones((1,2))))



def train_nn_surrogate_model(cfg: DictConfig, D, model: nn.Module, num_folds: int, rng_key: random.PRNGKey=jax.random.PRNGKey(0), model_type='regressor') -> float:
    # Split data into folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_indices = kf.split(D.X)


    # Train and validate the model for each fold
    fold_losses = []
    for train_index, val_index in fold_indices:
        # Split data into train and validation sets
        X_train, X_val = D.X[train_index], D.X[val_index]
        y_train, y_val = D.y[train_index], D.y[val_index]

        trained_params, _, _ = train(cfg, model, Dataset(X_train, y_train), Dataset(X_val, y_val), model_type=model_type)

        # Evaluate the model on the validation set
        y_pred = model.apply(trained_params, X_val)
        if model_type == 'classifier':
            fold_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), y_val))
        elif model_type == 'regressor':
            fold_loss = jnp.mean(jnp.square(y_val - y_pred))
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        fold_losses.append(fold_loss)

    # Compute average validation loss across folds
    avg_loss = jnp.mean(jnp.array(fold_losses))

    return avg_loss

       
class NeuralNetworkEstimator(nn.Module):
    hidden_units: list
    output_units: int
    activation_functions: list

    def setup(self):
        self.layers = [nn.Dense(hidden_unit) for hidden_unit in self.hidden_units] + [nn.Dense(self.output_units)]

        self.afs = []
        for i, af in enumerate(self.activation_functions):
            if af == 'relu':
                self.afs += (nn.relu,)
            elif af == 'sigmoid':
                self.afs += (nn.sigmoid,)
            elif af == 'tanh':
                self.afs += (nn.tanh,)

        self.afs += (lambda x: x,) # currently have no output activation function (feel free to change it)
        


    def __call__(self, x):
        for i, (layer, af) in enumerate(zip(self.layers, self.afs)):
            x = layer(x)
            x = af(x)

        return x 


def train_one_step_regressor(state, model, batch):
    @jit
    def loss_fn(params):
        y_pred = model.apply(params, batch['X'])
        loss = jnp.mean(jnp.square(batch['y'] - y_pred))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    return loss, grad


def train_one_step_classifier(state, model, batch):

    @jit
    def loss_fn(params):
        #labels must be a one hot encoded array
        y_pred = model.apply(params, batch['X'])
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), batch['y'] )) #  turns labels to 0 and 1, rather than -1 and 1, with zero indicating the negative class and 1 the positive class
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    
    return loss, grad


def get_initial_params(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  input_dims = tuple(data.X.shape[1:])  # (minibatch, height, width, stacked frames))
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)#['params']
  return initial_params

def get_initial_params_serial(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  input_dims = tuple(data.shape[1:])  # (minibatch, height, width, stacked frames))
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)#['params']
  return initial_params


def train(cfg, model, data, valid_data, model_type):

    # define optimizer
    if cfg.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=cfg.learning_rate,
            end_value=cfg.terminal_lr,
            transition_steps=cfg.num_epochs,
        )
    else:
        lr = cfg.learning_rate

    tx = optax.adam(lr)

    # initialise parameters
    params = get_initial_params(jax.random.PRNGKey(0), data, model)

    # create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    #state = jax_utils.replicate(state)

    # early stopping
    early_stop = EarlyStopping(min_delta=cfg.min_delta, patience=cfg.patience)

    loss_history = []  # Track loss history

    if model_type == 'regressor':
        train_one_step = train_one_step_regressor
    elif model_type == 'classifier':
        train_one_step = train_one_step_classifier
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    # Define a function to train one epoch
    def train_one_epoch(state, minibatch):
        loss, grads = train_one_step(state, model, minibatch)
        return loss, grads
    
    # Create minibatches
    num_devices = jax.local_device_count('cpu')
    minibatches = create_minibatches(data, cfg.batch_size, num_devices)
    minibatches = minibatch_reshape(minibatches)

    # Make the function parallelizable
    @partial(jax.pmap, axis_name="device", devices=[dev for i, dev in enumerate(jax.devices('cpu')) if i < minibatches['X'].shape[0]], in_axes=(None, 0), out_axes=(0, 0))
    def parallel_train_one_epoch(state, minibatches):

        # Train one epoch in parallel
        loss, grads = train_one_epoch(state, minibatches)
    
        # sync gradients
        grads = jax.lax.pmean(grads, "device")
        state = state.apply_gradients(grads=grads)
        loss = jax.lax.pmean(loss, "device")

        return state, loss
    
    # Train the model
    for epoch in range(cfg.num_epochs):

        # Train one epoch in parallel
        state, loss = parallel_train_one_epoch(state, minibatches)

        logging.info('epoch: %d, loss: %.4f' % (epoch, jnp.mean(loss).squeeze()))

        # Add current losses to history
        loss_history.extend(loss)

        # update kernel state
        state = jax_utils.unreplicate(state)
        

        # # NOTE removed validation data evaluation to hack around pmap (should be resolved) Evaluate the model on the validation data
        if model_type == 'classifier':
            val_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(model.apply(state.params, valid_data.X), axis=1), valid_data.y))
        elif model_type == 'regressor':
            val_loss = jnp.mean(jnp.square(valid_data.y - model.apply(state.params, valid_data.X)))
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        logging.info('Validation loss: %.4f' % val_loss)

        # Check for convergence
        early_stop = early_stop.update(val_loss)
        if isinstance(early_stop, tuple): early_stop = early_stop[1]
        if early_stop.should_stop:
            logging.info('Converged. Training stopped at iteration %d, loss value %.4f, val. loss value %.4f' % (epoch, jnp.mean(loss).squeeze(), val_loss))
            break

    if model_type == 'classifier':
        # get classifier performance
        def model_predict(params, data_points):
            return jnp.argmax(model.apply(params, data_points), axis=-1)
        
        def score(params, data_points, labels):
            return jnp.mean(jnp.equal(model_predict(params, data_points), labels.reshape(-1,)))
        
        accuracy = score(state.params, data.X, data.y)
        # get classifier false positive rates
        tn, fp, fn, tp = confusion_matrix(y_pred=jnp.astype(model_predict(state.params, data.X), jnp.int32), y_true=jnp.astype(data.y, jnp.int32).squeeze()).ravel()
        # metrics compression
        training_performance = {"acc": accuracy, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        logging.info(f"--- {model_type} ---")
        logging.info(f"training_performance: {training_performance}")

    if model_type == 'regressor':
        # get regressor performance
        mse = jnp.mean(jnp.square(data.y - model.apply(state.params, data.X)))
        training_performance = {"mse": mse, "standardised_mape": jnp.mean(jnp.abs((data.y - model.apply(state.params, data.X)) / data.y))}
        logging.info(f"--- {model_type} ---")
        logging.info(f"training_performance: {training_performance}")


      
    return state.params, model, loss_history


def get_serial_state_params(params):
    return {'params': {layer: {mod: val[0] for mod,val in layer_v.items() } for layer, layer_v in params['params'].items()}}


def minibatch_reshape(batches):
    return {'X': jnp.vstack([batch['X'].reshape(1,-1,batch['X'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]]), 'y': jnp.vstack([batch['y'].reshape(1,-1,batch['y'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]])}

def create_minibatches(dataset, batch_size, num_devices=1):
    num_examples = dataset.X.shape[0]
    num_batches = num_examples // batch_size

    if num_batches > num_devices:
        num_batches = num_devices
        if num_devices == 1:
            batch_size = num_examples
        else:
            batch_size = num_examples // (num_devices-1)

    minibatches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        if end > num_examples-1:
            end = num_examples-1
        minibatch = {'X': dataset.X[start:end], 'y': dataset.y[start:end]}
        minibatches.append(minibatch)


    # If there are leftover examples, create an additional mini-batch
    if num_examples % batch_size != 0:
        if num_devices == 1:
            start = num_batches * batch_size
            minibatch = {'X': dataset.X[start:], 'y': dataset.y[start:]}
            minibatches.append(minibatch)
        else:
            # upsample
            start = num_batches * batch_size
            num_remaining = start - (batch_size - (num_examples % batch_size))
            minibatch = {'X': dataset.X[num_remaining:], 'y': dataset.y[num_remaining:]}
            minibatches.append(minibatch)

    return [minibatches[batch] for batch in range(min(num_devices, len(minibatches)))]


def build_ann(cfg, model_data, model_class):

    # Determine the input and output dimensions
    x_standardisation = model_data['standardisation_metrics_input']
    x_mean = x_standardisation.mean
    x_std = x_standardisation.std

    # Create the neural network estimator
    model = NeuralNetworkEstimator(hidden_units=model_data['hidden_units'], output_units=model_data['output_units'], activation_functions=model_data['activation_function'])
    params = get_initial_params_serial(jax.random.PRNGKey(0), x_mean.reshape(1,-1), model)
    params = from_bytes(params, model_data['serialized_params'])
    opt_model = partial(model.apply, params)

    if model_class == 'regressor':
        y_standardisation = model_data['standardisation_metrics_output']
        y_mean = y_standardisation.mean
        y_std = y_standardisation.std

        y_mean = jnp.array(y_mean)
        y_std = jnp.array(y_std)

        @jit
        def project(y):
            return y * y_std + y_mean
        
        if cfg['solvers']['standardised']:
            @jit
            def query_standardised_model(x):
                if x.ndim <2: x = x.reshape(1,-1)
                return opt_model(x)
            return query_standardised_model    

        else:
            @jit
            def standardise(x):
                return (x - x_mean) / x_std
            @jit
            def query_unstandardised_model(x):
                if x.ndim <2 : x = x.reshape(1,-1)
                return project(opt_model(standardise(x)))
            return query_unstandardised_model

    
    elif model_class == 'classifier':

        def mapp_(y):
            if y.ndim <2 : y = y.reshape(1,-1)
            return jnp.array([0.5]) - softmax(y, axis=-1)[0,0]    # if this quantity is less than or equal to zero, then the sample predicts feasibility.

        if cfg['solvers']['standardised']:
            @jit
            def query_standardised_classifier(x):
                if x.ndim <2: x = x.reshape(1,-1)
                if x.shape[0]>= x.shape[1]: x= x.T
                return mapp_(opt_model(x))
            return query_standardised_classifier
        
        else:
            @jit
            def standardise(x):
                return (x - x_mean) / x_std
            @jit
            def query_unstandardised_classifier(x):
                if x.ndim <2 : x = x.reshape(1,-1)
                return mapp_(opt_model(standardise(x))) 
    
            
            return query_unstandardised_classifier
    
    else:
        raise NotImplementedError(f"Model class {model_class} not implemented")

    