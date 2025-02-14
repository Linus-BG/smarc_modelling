import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mse(filepath):

    real_data = pd.read_csv("src/smarc_modelling/pinn/data/system_states.csv") # TODO: Replace with real data instead of model collected
    state_columns = ["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "p", "q", "r"]
    real_time = real_data["Time"].values
    real_states = real_data[state_columns].values

    gen_data = pd.read_csv(filepath)
    gen_states = gen_data[state_columns].values

    pinn_mse = np.mean((gen_states - real_states)**2, axis=1)
    return pinn_mse, real_time

# Plot results
def plot():
    plt.figure(figsize=(10, 5))
    plt.plot(real_time, model_mse, label='Model MSE', color='blue')
    plt.plot(real_time, pinn_mse, label='PINN MSE', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('Model vs PINN MSE')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Compute Mean Squared Error (MSE)
    model_mse, _ = get_mse("src/smarc_modelling/pinn/data/system_states_spin.csv")
    pinn_mse, real_time = get_mse("src/smarc_modelling/pinn/data/pinn_results.csv")
    plot()

