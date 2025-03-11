from smarc_modelling.piml.pigp.pigp import MultiTaskGP
from smarc_modelling.piml.pigp.pigp import DeepKernelNN
from smarc_modelling.piml.prepare_data import prepare_data
import gpytorch
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Params and dummies
    num_tasks = 6
    x_dummy = torch.empty(0, 19)

    # Init model and likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    feature_network = DeepKernelNN(64)
    model = MultiTaskGP(x_dummy, None, likelihood, feature_network)

    # Load weights and turn on eval
    likelihood.load_state_dict(torch.load("src/smarc_modelling/piml/models/gp.pt", weights_only=True)["likelihood"])
    model.load_state_dict(torch.load("src/smarc_modelling/piml/models/gp.pt", weights_only=True)["model"])
    feature_network.load_state_dict(torch.load("src/smarc_modelling/piml/models/gp.pt", weights_only=True)["featurenet"])
    likelihood.eval()
    model.eval()
    feature_network.eval()

    # Loading data for predictions 
    # eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data("src/smarc_modelling/piml/data/system_states_validate.csv", "torch")
    eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data("src/smarc_modelling/piml/data/system_states_spin_and_straight.csv", "torch")
    x_vec = torch.cat([eta, nu, u], dim=1) 
    num_data = eta.shape[0]

    # Pre-allocating space for predictions
    Dv_pred = torch.empty(num_data, 6)
    lower_bounds = torch.empty(num_data, 6)
    upper_bounds = torch.empty(num_data, 6)

    print(f" Making predictions...")
    for t in range(num_data):
        x_curr = x_vec[t].unsqueeze(0)
        with torch.no_grad():#, gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(x_curr))
            mean = prediction.mean
            lower, upper = prediction.confidence_region()
        Dv_pred[t] = mean
        lower_bounds[t] = lower
        upper_bounds[t] = upper
    print(f" Prediction done!")

    num_data_vec = range(num_data)

    plt.figure(figsize=(12,8))
    plt.suptitle("Estimations vs GT")
    for i in range(num_tasks):
        plt.subplot(6, 1, i + 1)
        plt.scatter(
            num_data_vec, Dv_comp[:, i].detach().numpy(), s=1, color="orange", label="Calculated damping force"
        )
        plt.scatter(
            num_data_vec, Dv_pred[:, i].detach().numpy(), s=1, color="blue", label="Predicted damping force"
        )
        plt.fill_between(
            num_data_vec, 
            lower_bounds[:, i].detach().numpy(), 
            upper_bounds[:, i].detach().numpy(), 
            color="blue", 
            alpha=0.2, 
            label="95% Confidence Interval"
        )
        if i == 0:
            plt.legend()

    plt.show()