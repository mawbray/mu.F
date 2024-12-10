import torch
import torch.optim as optim
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np
import sobol_seq
import math

import logging


def generate_sobol_points(lower_bound, upper_bound, num_points=10):
    lower_bound = np.array(lower_bound)  # Convert to numpy array
    upper_bound = np.array(upper_bound)  # Convert to numpy array

    dim = len(lower_bound)  # Ensure Sobol matches the correct dimensionality
    sample = sobol_seq.i4_sobol_generate(dim, num_points)

    # Scale samples to the desired range
    points = lower_bound + (upper_bound - lower_bound) * sample
    return points

class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def bayesian_optimization(f, lower_bound, upper_bound, num_initial_points, num_iterations):
    logging.info("Starting Bayesian optimization...")
    logging.info(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    logging.info(f"Num initial points: {num_initial_points}, Num iterations: {num_iterations}")

    # Generate Sobol points: now treat each as a different candidate
    train_x = torch.tensor(generate_sobol_points(lower_bound, upper_bound, num_initial_points)).float().T

    logging.info(f"Shape of train_x: {train_x.shape}")  # Print the shape of train_x for debugging
    logging.info(f"x-values: {train_x}")  # Print the x-values for debugging

    # Ensure correct dimensionality of train_x
    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    logging.info(f"Shape of train_x: {train_x.shape}")

    # Evaluate the objective function for each candidate in train_x
    train_y = torch.vstack([f(train_x[:, i]).reshape(1,1) for i in range(train_x.shape[1])]).squeeze()

    likelihood = GaussianLikelihood()
    

    #train_x = (train_x - train_x.mean()) / train_x.std()
    #train_y = (train_y - train_y.mean()) / train_y.std()

    model = GPRegressionModel(train_x.T, train_y, likelihood)

    fit_gpytorch_model(likelihood, model, train_x.T, train_y)

    logging.info('Initial min:', train_y.min().item())

    for _ in range(num_iterations):
        candidate_x = select_next_point(model, lower_bound, upper_bound)
        #print(f"Shape of candidate_x: {candidate_x.shape}")
        candidate_y = f(candidate_x).reshape(-1,)

        train_x = torch.cat([train_x, candidate_x.unsqueeze(1)], dim=1)
        train_y = torch.cat([train_y, candidate_y])

        model.set_train_data(train_x.T, train_y, strict=False)

        # Reinitialize and fit the model
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(train_x.T, train_y, likelihood)
        #here
        fit_gpytorch_model(likelihood, model, train_x.T, train_y)

    # Find the best candidate (minimum y value)
    best_index = torch.argmin(train_y)
    best_candidate = train_x[:, best_index]

    logging.info(f'Final min: {train_y.min().item()}')
    logging.info(f'Best candidate: {best_candidate}')

    torch.save(train_y, 'opt_y.pt')
    torch.save(train_x, 'opt_x.pt')

    return best_candidate, best_index  # Return the best candidate, which will be used to update C




def original_bayesian_optimization(f, lower_bound, upper_bound, num_initial_points, num_iterations):
    train_x = torch.tensor(generate_sobol_points(lower_bound, upper_bound, num_initial_points)).float().T

    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    logging.info(f"Shape of train_x: {train_x.shape}")  # Print the shape of train_x for debugging

    train_x = train_x[:, 0]  # Take the first candidate for simplicity TODO: check

    train_y = f(train_x).squeeze()
    logging.info(f"Shape of train_y: {train_y.shape}")  # Add this to check the shape

    likelihood = GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    logging.info(f"Shape of train_x: {train_x.shape}, Shape of train_y: {train_y.shape}")

    fit_gpytorch_model(likelihood, model, train_x, train_y)

    logging.info('initial min:', train_y.min().item())

    for _ in range(num_iterations):
        candidate_x = select_next_point(model, lower_bound, upper_bound)
        candidate_y = f(candidate_x).reshape(-1,)

        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, candidate_y])

        model.set_train_data(train_x, train_y, strict=False)

        # create a new model and fit it NOTE there is a better way to do this, by reinitializing the model
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood)
        fit_gpytorch_model(likelihood, model, train_x, train_y)

    logging.info('final min:', train_y.min().item())

    return model

def fit_gpytorch_model(likelihood, model, train_x, train_y):
    optimizer = optim.LBFGS(model.parameters(), lr=0.0001, line_search_fn='strong_wolfe')
    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, model)

    tolerance = 1e-6  # Define your tolerance threshold
    previous_loss = None
    for k in range(1000):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y.squeeze())
            loss.backward()
            return loss
        
        optimizer.step(closure)

        loss = -mll(model(train_x), train_y).item()
        
        # Check for convergence
        if previous_loss is not None:
            change_in_loss = abs(loss - previous_loss)
            if change_in_loss <= tolerance:
                logging.info(f"Convergence achieved. after {k+1} iterations.")
                break  # Exit the loop if convergence is achieved
        
        # Update the previous_loss for the next iteration
        previous_loss = loss
    likelihood.eval()
    model.eval()

def select_next_point(model, lower_bound, upper_bound, acq:str='ucb'):
    # Implement your acquisition function here
    # For example, you can use Expected Improvement (EI)
    # EI = (mu - f_best) * Phi(Z) + sigma * phi(Z)
    # where mu is the predicted mean, f_best is the best observed value so far,
    # sigma is the predicted standard deviation, Z = (mu - f_best) / sigma,
    # Phi is the cumulative distribution function of the standard normal distribution,
    # and phi is the probability density function of the standard normal distribution.
    # Return the point with the maximum acquisition value.

    if acq == 'ucb':
        return ucb(model, lower_bound, upper_bound)
    else:
        raise ValueError(f"Invalid acquisition function: {acq}")


def ucb(model, lower_bound, upper_bound):
    # Ensure the bounds are tensors of shape [26]
    bounds = torch.stack([lower_bound, upper_bound], dim=1)  # Shape [26, 2]

    # Define the acquisition function (Upper Confidence Bound)
    def acquisition(x):
        model.eval()
        with gpytorch.settings.fast_pred_var():
            # Reshape x to match the expected input size (should be [1, 26])
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Ensure x has shape [1, 26]

            # Predict mean and variance for x
            pred = model(x)
            mean = pred.mean
            variance = pred.variance

            # UCB formula: mean + beta * variance
            return - (mean + 1.96 * torch.sqrt(variance))  # Maximize UCB (negative for minimizing)

    # Initial candidate point
    x0 = (lower_bound + upper_bound) / 2
    x0 = x0.clone().detach().requires_grad_(True)

    # Use LBFGS optimizer to optimize the acquisition function
    optimizer = torch.optim.LBFGS([x0], max_iter=20)

    def closure():
        optimizer.zero_grad()
        loss = acquisition(x0)
        loss.backward()
        return loss

    # Perform optimization step
    optimizer.step(closure)

    # Ensure x0 stays within bounds after optimization by clipping
    x0 = torch.clamp(x0, lower_bound, upper_bound)

    # Check the shape after optimization (should be [26])
    logging.info(f"Shape of candidate_x after optimization: {x0.shape}")
    return x0.detach()



def original_ucb(model, lower_bound, upper_bound):
    # Implement the Upper Confidence Bound (UCB) acquisition function


    def ucb_acquisition(x):
        # Implement the UCB acquisition function
        # UCB = mu + beta * sigma
        # where mu is the predicted mean, sigma is the predicted standard deviation,
        # and beta is a hyperparameter that controls the trade-off between exploration and exploitation. 

        # this is defined to optimistically select minimas
        pred  = model(x)
        beta = 2
        return pred.mean - beta * pred.stddev
    
    

    # Create the LBFGS optimizer
    n_multi_start = 10
    n_grad_steps = 50
    points = generate_sobol_points(lower_bound, upper_bound, n_multi_start).T
    minima = []
    sol_x  = []
    tolerance = torch.tensor([1e-8])  # Define your tolerance threshold
    previous_acquisition_value = None

    for i in range(n_multi_start):
        next_point = torch.tensor(points[i].reshape(1,-1)).float().requires_grad_(True)
        optimizer = optim.LBFGS([next_point], line_search_fn='strong_wolfe', lr=0.01)

        # Optimize the UCB acquisition function
        for j in range(n_grad_steps):
            # Define the closure function for LBFGS optimizer
            def closure():
                optimizer.zero_grad()
                acquisition_value = ucb_acquisition(next_point)
                acquisition_value.backward()
                return acquisition_value
            
            optimizer.step(closure)
            with torch.no_grad(): next_point[:] = next_point.clamp(lower_bound, upper_bound)

            # Get the current acquisition value
            current_acquisition_value = ucb_acquisition(next_point).float().detach().clone().requires_grad_(False)

            # Check for convergence
            if previous_acquisition_value is not None:
                change_in_value = torch.absolute(current_acquisition_value - previous_acquisition_value)
                if change_in_value <= tolerance:
                    print(f"Convergence achieved after {j+1} iterations.")
                    break  # Exit the loop if convergence is achieved

            # Update the previous_acquisition_value for the next iteration
            previous_acquisition_value = current_acquisition_value

        minima.append(current_acquisition_value)
        sol_x.append(next_point.clone().detach())

    # Select the point with the minimum acquisition value
    minima = torch.tensor(minima)
    min_idx = torch.argmin(minima)
    next_point = sol_x[min_idx]
    return next_point