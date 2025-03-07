#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gpytorch
from smarc_modelling.piml.prepare_data import prepare_data


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Mean and Kernel TODO: Design kernel for physics loss
        self.mean_module =  gpytorch.means.LinearMean(train_x.shape[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":

    # Load and prepare data
    eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data("src/smarc_modelling/piml/data/system_states_spin_and_straight.csv", "torch")
    train_x = torch.cat([eta, nu, u], dim=1)  # [6000, 19]
    num_data = eta.shape[0]
    
    # Each task is a value in D(v)v
    num_tasks = 6

    # Create a separate GP and likelihood for each task
    models = []
    likelihoods = []
    # Give each task its own distribution to be trained
    for i in range(num_tasks):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(train_x, Dv_comp[:, i], likelihood) # Picks out one element to train from
        models.append(model)
        likelihoods.append(likelihood)

    # Set training mode for each model
    for model, likelihood in zip(models, likelihoods):
        model.train()
        likelihood.train()

    # Define optimizers and loss functions for each GP
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.1) for model in models]
    mlls = [gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) for likelihood, model in zip(likelihoods, models)]

    # Training loop
    epochs = 500
    print(f" Starting training...")
    for epoch in range(epochs + 1):
        loss_total = 0

        for i in range(num_tasks):
            optimizers[i].zero_grad()
            output = models[i](train_x)  # Get prediction
            loss = -mlls[i](output, Dv_comp[:, i])  # Compute loss
            loss.backward()
            optimizers[i].step()
            loss_total += loss.item()

        if loss_total < 10.0:
            print(f" Stopping training early at epoch: {epoch}, loss: {loss_total}")
            break 

        if epoch % 10 == 0:
            print(f" Epoch: {epoch}, Total Loss: {loss_total}")
        
    
    print(f" Training complete saving model & likelihood to gp.pt")

    # Save models and likelihoods
    state_dicts = {
        "models": [model.state_dict() for model in models],
        "likelihoods": [likelihood.state_dict() for likelihood in likelihoods]
    }

    torch.save(state_dicts, "src/smarc_modelling/piml/models/gp.pt")

    
    x_test_eta = eta[-50]  
    x_test_nu = nu[-50]    
    x_test_u = u[-50]     
    x_test_tensor = torch.cat([x_test_eta, x_test_nu, x_test_u], dim=0)
    x_test_tensor = x_test_tensor.unsqueeze(0)  # [1, 19]

    # Loop through each model and compute the predictions
    real_dv = []
    predicted_dv= []
    for i in range(num_tasks):
        model = models[i]
        likelihood = likelihoods[i]

        # Set the model and likelihood to eval mode
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            # Get the prediction
            observed_pred = likelihood(model(x_test_tensor))
            predicted_mean = observed_pred.mean.item()  # Get the predicted mean

            predicted_dv.append(predicted_mean)
            real_dv.append(Dv_comp[-50, i].item())

    print(f" Real D(v)v: {real_dv}, \n Predicted D(v)v: {predicted_dv}")