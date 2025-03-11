#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gpytorch
from smarc_modelling.piml.prepare_data import prepare_data

class DeepKernelNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(DeepKernelNN, self).__init__()
        self.fc1 = torch.nn.Linear(19, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 19)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc_out(x)
        return x

class MultiTaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_network):
        super(MultiTaskGP, self).__init__(train_x, train_y, likelihood)

        self.feature_network = feature_network
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.,1.) # https://docs.gpytorch.ai/en/v1.13/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(train_x.shape[-1]), num_tasks=6
            # gpytorch.means.ConstantMean(), num_tasks=6
        )

        base_kernel = gpytorch.kernels.LinearKernel() + gpytorch.kernels.MaternKernel(nu=1.5)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=6, rank=1
        )

    def forward(self, x):
        transformed_x = self.feature_network(x)
        transformed_x = self.scale_to_bounds(transformed_x) # Makes the NN values "Nice"
        mean_x = self.mean_module(transformed_x)
        covar_x = self.covar_module(transformed_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":

    # Load and prepare data
    eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data("src/smarc_modelling/piml/data/system_states_spin_and_straight.csv", "torch")

    # # Normalize data for better training
    # eta_norm = (eta - eta.mean()) / eta.std()
    # nu_norm = (nu - nu.mean()) / nu.std()
    # u_norm = (u - u.mean()) / u.std()
    # Dv_comp_norm = (Dv_comp - Dv_comp.mean()) / Dv_comp.std()

    # Set up training data
    train_x = torch.cat([eta, nu, u], dim=1)  # [6000, 19]
    train_y = Dv_comp
    num_data = eta.shape[0]
    num_tasks = 6

    feature_network = DeepKernelNN(hidden_dim=64)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6)
    model = MultiTaskGP(train_x, train_y, likelihood, feature_network)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
    {'params': model.feature_network.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
    ], lr=0.05)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    print(f" Starting training...")
    for i in range(100):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, 100, loss.item()))
        optimizer.step()

        if loss.item() < 0:
            print(f" Ending training early")
            break

    
    print(f" Training complete saving model & likelihood to gp.pt")

    # # Normalization stuff for prediction
    # train_x_mean = train_x.mean(dim=0)
    # train_x_std = train_x.std(dim=0)

    # Save models and likelihoods
    model_dict = {
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),
        "featurenet": feature_network.state_dict()
    }

    torch.save(model_dict, "src/smarc_modelling/piml/models/gp.pt")

    x_test_eta = eta[-50]  
    x_test_nu = nu[-50]    
    x_test_u = u[-50]     
    x_test_tensor = torch.cat([x_test_eta, x_test_nu, x_test_u], dim=0)
    x_test_tensor = x_test_tensor.unsqueeze(0)  # [1, 19]

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        predicted_dv = likelihood(model(x_test_tensor)).mean
        real_dv = Dv_comp[-50, :]

    print(f" Real D(v)v: {real_dv}, \n Predicted D(v)v: {predicted_dv}")