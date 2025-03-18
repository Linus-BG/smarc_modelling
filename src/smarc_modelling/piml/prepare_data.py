#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from smarc_modelling.vehicles.SAM import SAM

def prepare_data(data_file="", return_type=""):

    """
    Calculates D * nu.
    At the moment all data is from the model but in the future v_dot_ord(inary) will be computed
    from the collected data and v_dot_nod(No-D) will be computed with Fossen without dampening
    from the collected data.
    """

    # Define the control reference input 
    # NOTE: Later this will not be needed when we have the real data just the actual current actuator info
    u_ref = np.zeros(6)
    u_ref[0] = 50#*np.sin((i/(20/0.02))*(3*np.pi/4))        # VBS
    u_ref[1] = 50 # LCG
    u_ref[2] = np.deg2rad(7)    # Vertical (stern)
    u_ref[3] = -np.deg2rad(7)   # Horizontal (rudder)
    u_ref[4] = 1000     # RPM 1
    u_ref[5] = u_ref[4]     # RPM 2

    print(f" Loading data")
    # Load the generated data
    data = pd.read_csv(data_file)
    # Pulling out individual stuff from the data
    time = data["Time"].values
    states = data[["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "p", "q", "r"]].values
    controls = data[["VBS", "LCG", "DS", "DR", "RPM1", "RPM2"]].values
    inputs = np.concatenate([states, controls], axis=1)

    # Initialise the SAM white-box model
    sam = SAM(dt=0.01)

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
    if return_type == "torch":
        return(
            torch.tensor(eta, dtype=torch.float32),
            torch.tensor(nu, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(Dv_comp, dtype=torch.float32),
            torch.tensor(Mv_dot, dtype=torch.float32),
            torch.tensor(Cv, dtype=torch.float32),
            torch.tensor(g_eta, dtype=torch.float32),
            torch.tensor(tau, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32)
        )
    else:
        return(
            eta,
            nu,
            u,
            Dv_comp,
            Mv_dot,
            Cv,
            g_eta,
            tau,
            time
        )
