import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Convert quaternion to roll, pitch, yaw
def quaternions_to_euler(states):
    quaternions = states[:, 3:7]  # Extract q0, q1, q2, q3
    eulers = R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)  # Convert to roll, pitch, yaw
    return np.hstack((states[:, :3], eulers, states[:, 7:]))  # Ensure correct slicing


# Loading ground truth data NOTE: Currently the same as model data(!)
ground_truth = pd.read_csv("src/smarc_modelling/piml/data/system_states_validate.csv")
times = ground_truth["Time"].values
state_vec = ["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "p", "q", "r"]
gt_states = ground_truth[state_vec].values

# Loading model results
model_result = pd.read_csv("src/smarc_modelling/piml/data/system_states_validate.csv")
model_states = model_result[state_vec].values

# Loading PINN results
pinn_result = pd.read_csv("src/smarc_modelling/piml/data/pinn_results.csv")
pinn_states = pinn_result[state_vec].values

# Loading BPINN mean results
bpinn_resuts = pd.read_csv("src/smarc_modelling/piml/data/bpinn_results.csv")
bpinn_states = bpinn_resuts[state_vec].values

# Convert states to Euler angles
gt_states = quaternions_to_euler(gt_states)
model_states = quaternions_to_euler(model_states)
pinn_states = quaternions_to_euler(pinn_states)
bpinn_states = quaternions_to_euler(bpinn_states)

# Update state vector
state_vec = ["x (m)", "y (m)", "z (m)", "roll (°)", "pitch (°)", "yaw (°)", "u (m/s)", "v (m/s)", "w (m/s)", "p (°/s)", "q (°/s)", "r (°/s)"]

# Error calculation
model_error = model_states - gt_states
pinn_error = pinn_states - gt_states
bpinn_error = bpinn_states - gt_states

# Cumulative MSE error calculation
cumulative_model_mse = np.cumsum(model_error ** 2, axis=0)
cumulative_pinn_mse = np.cumsum(pinn_error ** 2, axis=0)
cumulative_bpinn_mse = np.cumsum(bpinn_error ** 2, axis=0)

# Plot results directly
plt.figure(figsize=(12, 8))
plt.suptitle("Predictions of state")
for i, state in enumerate(state_vec):
    plt.subplot(4, 3, i + 1)
    plt.plot(times, model_states[:, i], linestyle='-', label="Model State")
    plt.plot(times, pinn_states[:, i], linestyle='-', label="PINN State")
    plt.plot(times, bpinn_states[:, i], linestyle="-", label="B-PINN State")
    plt.plot(times, gt_states[:,i], linestyle="-", label="Ground Truth [Model]")
    plt.xlabel("Time (s)")
    plt.ylabel(state)
    plt.grid()

plt.legend(loc='upper right', bbox_to_anchor=(1, 5.1), ncol=2)
plt.tight_layout(h_pad=10)

# Plot cumulative MSE error
plt.figure(figsize=(12, 8))
plt.suptitle("Cumulative MSE Error")
for i, state in enumerate(state_vec):
    plt.subplot(4, 3, i + 1)
    plt.plot(times, cumulative_model_mse[:, i], linestyle='-', label="Model MSE")
    plt.plot(times, cumulative_pinn_mse[:, i], linestyle='-', label="PINN MSE")
    plt.plot(times, cumulative_bpinn_mse[:, i], linestyle="-", label="B-PINN MSE [Mean]")
    plt.xlabel("Time (s)")
    plt.ylabel(state)
    plt.grid()

plt.legend(loc='upper right', bbox_to_anchor=(1, 5.1), ncol=2)
plt.tight_layout(h_pad=10)

# Plot direct error
plt.figure(figsize=(12, 8))
plt.suptitle("Error")
for i, state in enumerate(state_vec):
    plt.subplot(4, 3, i + 1)
    plt.plot(times, model_error[:, i], linestyle='-', label="Model Error")
    plt.plot(times, pinn_error[:, i], linestyle='-', label="PINN Error")
    plt.plot(times, bpinn_error[:, i], linestyle="-", label="B-PINN Error [Mean]")
    plt.xlabel("Time (s)")
    plt.ylabel(state)
    plt.grid()



plt.legend(loc='upper right', bbox_to_anchor=(1, 5.1), ncol=2)
plt.tight_layout(h_pad=10)
plt.show()
