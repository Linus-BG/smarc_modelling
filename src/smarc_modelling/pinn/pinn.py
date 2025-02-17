#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from smarc_modelling.vehicles.SAM import SAM

# Functions and classes
class PINN(nn.Module):

    """
    Physics informed neural network that currently enforces:
    1. Symmetry in dampening matrix by training A such that A*A^{T} = D (strict)
    2. The above thing also ensures that the matrix will be positive semi-definite (strict)
    3. Diagonal elements are also strictly positive from point 1 (strict)
    4. Fossen equation should be upheld. Done trough loss function (suggested)
    """

    def __init__(self):
        super(PINN, self).__init__()
        # TODO: Just some random layers atm, check more into this later
        # Defining the layers
        self.fc1 = nn.Linear(19, 32) # Neural network takes all 19 inputs for prediction
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 36)

        # 38180 params -> Should have 10x data points (6000 atm...)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # TODO: Check different activation functions
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        A_flat = self.fc6(x)
        A_mat = A_flat.view(-1, 6, 6) # Go from just a row of values to matrix
        D = A_mat @ A_mat.transpose(-2, -1) # Enforce symmetry of D matrix
        return D
    

def loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu):
    """
    Computes the physics-informed loss by comparing NN output with expected calculations
    Sums this with the data loss (Dv_comp - Dv_pred)

    """
    
    # Getting the current prediction for D
    D_pred = model(x)
    
    # Calculate physics loss 
    # Enforce Fossen model
    physics_loss = torch.mean((Mv_dot + Cv + (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)) + g_eta - tau)**2)

    # Calculate data loss
    data_loss = torch.mean((Dv_comp - (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)))**2)

    # L1 norm to encourage sparsity / parsimony  <-- Actually this does not encourage sparsity or parsimony as it only affects the NN structure not the D matrix
    # NOTE: One way to actually do this would be summing all the elements in the matrix and presenting that as a loss
    l1_norm = 0 # sum(p.abs().sum() for p in model.parameters())

    loss = physics_loss + data_loss + l1_norm # We value learning the physics over just fitting the data

    return loss


def prepare_data(inputs, sam, u_ref):

    """
    Calculates D * nu and converts data to torch tensors.
    At the moment all data is from the model but in the future v_dot_ord(inary) will be computed
    from the collected data and v_dot_nod(No-D) will be computed with Fossen without dampening
    from the collected data.
    """

    print(f" Preparing data...")

    # Pulling out the velocities
    # NOTE: Replace with real data later
    eta = inputs[:, 0:7]        # Position and orientation
    nu = inputs[:, 7:13]        # Velocities
    u = inputs[:, 13:19]        # Control inputs NOTE: Should this be included in the NN, its not like the dampening depends on how fast our propellers are spinning(?)

    # Pre-allocating space for accelerations
    v_dot_ord = np.zeros_like(nu)
    v_dot_nod = np.zeros_like(nu)
    Dv_comp = np.zeros((len(inputs), 6))

    # These will be needed when computing the physics loss
    Mv_dot = np.zeros_like(nu)
    Cv = np.zeros_like(nu)
    g_eta = np.zeros_like(nu)
    tau = np.zeros_like(nu)

    print(f" Starting with spinning data set.")

    for t in range(len(inputs)):

        if t == 3000:
            print(f" Moving to straight data set.")
            u_ref[2] = 0
            u_ref[3] = 0

        # Computing the full model results (D included)
        v_dot_ord[t] = sam.dynamics(inputs[t], u_ref)[7:13] # This should be replaced with the real measured acceleration later
        
        # Computing the model dynamics where D = 0
        # NOTE: Currently the sam.dynamics calls to update the different matrices but if it is removed later a section calling sam.calculate_X will be needed
        v_dot_nod[t] = sam.Minv @ (sam.tau - sam.C @ nu[t] - sam.g_vec)

        # Calculate the self dampening elements based on the current speed
        Dv_comp[t] = sam.M @ (v_dot_nod[t].T - v_dot_ord[t].T)

        # Calculating misc things for physics loss
        Mv_dot[t] = sam.M @ v_dot_ord[t]
        Cv[t] = sam.C @ nu[t]
        g_eta[t] = sam.g_vec
        tau[t] = sam.tau

    print(f" Data preperation complete!")

    return(
        torch.tensor(eta, dtype=torch.float32),
        torch.tensor(nu, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
        torch.tensor(Dv_comp, dtype=torch.float32),
        torch.tensor(Mv_dot, dtype=torch.float32),
        torch.tensor(Cv, dtype=torch.float32),
        torch.tensor(g_eta, dtype=torch.float32),
        torch.tensor(tau, dtype=torch.float32)
    )


if __name__ == "__main__":

    # Define the control reference input 
    # NOTE: Later this will not be needed when we have the real data just the actual current actuator info
    u_ref = np.zeros(6)
    u_ref[0] = 50#*np.sin((i/(20/0.02))*(3*np.pi/4))        # VBS
    u_ref[1] = 50 # LCG
    u_ref[2] = np.deg2rad(7)    # Vertical (stern)
    u_ref[3] = -np.deg2rad(7)   # Horizontal (rudder)
    u_ref[4] = 1000     # RPM 1
    u_ref[5] = u_ref[4]     # RPM 2

    # TODO: Maybe generate data here so that it can be easier to change the reference control inputs for now

    # Load the generated data
    data = pd.read_csv("src/smarc_modelling/pinn/data/system_states_spin_and_straight.csv")
    # Pulling out individual stuff from the data
    time = data["Time"].values
    states = data[["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "p", "q", "r"]].values
    controls = data[["VBS", "LCG", "DS", "DR", "RPM1", "RPM2"]].values
    inputs = np.concatenate([states, controls], axis=1)

    # Initialize the SAM white-box model
    sam = SAM(dt=0.01)

    # Prepping the data
    eta, nu, u, Dv_comp, Mv_dot, Cv, g_eta, tau = prepare_data(inputs, sam, u_ref)
    x = torch.cat([eta, nu, u], dim=1)

    # Initialize PINN model & optimizer
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # TODO: Tune learning rate

    # Adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=500, threshold=0.01, min_lr=1e-5)

    # Training loop
    epochs = 50000
    print(f" Starting training...")
    for epoch in range(epochs):
        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        D_pred = model(x)

        # Calculate loss
        loss = loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step(loss)

        if epoch % 500 == 0:
            print(f" Still training, epoch: {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")

        # End early if our loss is small enough
        if loss.item() < 0.1: # TODO: Tune this
            print(f" Training stopped early due to reaching loss threshold at epoch: {epoch}")
            break
    
    # Saving NN
    torch.save(model.state_dict(), "src/smarc_modelling/pinn/models/pinn.pt")
    print(f"\n Model weights saved to models/pinn.pt")

    # Quickly testing results
    # Evaluation mode
    model.eval()

    # Test state
    x_test = x[-50]

    # Get predicted D(v)
    learned_D = model(x_test).detach().numpy()

    # Model prediction
    sam.dynamics(x_test.detach().numpy(), u_ref) # Calling this to update D using state x_test

    # Take-aways
    # 1. In scenarios where nu is low velocities / zero velocity the dampening matrix will look a bit weird as compared
    # to the model one but when multiplied with the nu again most weird terms become 0 anyways so the real dynamics are represented
    np.set_printoptions(precision=3, suppress=True)
    print(f" \n v: \n", nu[-10].detach().numpy())
    print(f" Learned D(v)v as: \n", learned_D @ nu[-10].detach().numpy())
    print(f" Model D(v)v as: \n", sam.D @ nu[-10].detach().numpy())
    print("\n")
    print(f" Learned D(v) as \n", learned_D)
    print(f" Model D(v) as: \n", sam.D)