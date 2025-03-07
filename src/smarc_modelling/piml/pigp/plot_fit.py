from smarc_modelling.piml.pigp.pigp import GP
from smarc_modelling.piml.prepare_data import prepare_data
import gpytorch
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load model
    models =  []
    likelihoods = []

    # Params and dummies
    tasks = 6
    x_dummy = torch.empty(0, 19)

    # Init models and likelihoods
    for i in range(tasks):
        
        # Create likelihood and model for each task
        likelihood = gpytorch.likelihoods.GaussianLikelihood().eval()
        model = GP(x_dummy, None, likelihood)
        model.eval()

        # Load the state dictionaries for the model and likelihood
        likelihood.load_state_dict(torch.load("src/smarc_modelling/piml/models/gp.pt", weights_only=True)["likelihoods"][i])
        model.load_state_dict(torch.load("src/smarc_modelling/piml/models/gp.pt", weights_only=True)["models"][i])
        
        # Append to the lists
        likelihoods.append(likelihood)
        models.append(model)

    # Loading data for predictions 
    eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data("src/smarc_modelling/piml/data/system_states_validate.csv", "torch")
    x_vec = torch.cat([eta, nu, u], dim=1) 
    num_data = eta.shape[0]

    # Pre allocatiing space for predictions
    Dv_pred = torch.empty(num_data, 6)

    print(f" Making predictions...")
    for t in range(num_data):
        
        # Getting current feature vector
        x_curr = x_vec[t].unsqueeze(0)
        
        with torch.no_grad():
            for i in range(tasks):
                # Pull out set for the task
                model = models[i]
                likelihood = likelihoods[i]

                # Make prediction 
                observed_pred = likelihood(model(x_curr))
                Dv_pred[t, i] = observed_pred.mean.item()
    print(f" Prediction done!")

    num_data_vec = range(num_data)

    plt.figure(figsize=(12,8))
    plt.suptitle("Estimations vs GT")
    for i in range(tasks):
        plt.subplot(6, 1, i+1)
        plt.plot(num_data_vec, Dv_comp[:, i].detach().numpy(), linestyle="-", label="Calculated damping force")
        plt.plot(num_data_vec, Dv_pred[:, i].detach().numpy(), linestyle="-", label="Predicted damping force")
    plt.legend()
    plt.show()