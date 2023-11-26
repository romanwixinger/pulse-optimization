""" Visualizes the result of main_real.py."""
import numpy as np
import matplotlib.pyplot as plt

# Settings
folder = "data/real/fourier_extended_constrained"

# Function to load data
def load_data(filename):
    return np.loadtxt(filename)

# Circuit lengths used in the experiment
circuit_lengths = range(0, 110, 10)  # From 0 to 100 with step 10

# Scaling factors used in the experiment
scaling_factors = np.arange(0.0, 1.0, step=0.1)

# Plot for r00 data
plt.figure(figsize=(10, 6))
for length in circuit_lengths:
    r00_data = [load_data(f'{folder}/r00_X_DEVICE_1_{scaling}.txt')[length // 10] for scaling in scaling_factors]
    plt.plot(scaling_factors, r00_data, label=f'Length {length}')
plt.title('r00 Data')
plt.xlabel('Scaling Factor')
plt.ylabel('Hellinger Distance')
plt.legend(fontsize='small', loc='upper right')
plt.grid(True)
plt.savefig("r00_data_visualization.svg")
plt.show()

# Plot for r11 data
plt.figure(figsize=(10, 6))
for length in circuit_lengths:
    r11_data = [load_data(f'{folder}/r11_X_DEVICE_1_{scaling}.txt')[length // 10] for scaling in scaling_factors]
    plt.plot(scaling_factors, r11_data, label=f'Length {length}')
plt.title('r11 Data')
plt.xlabel('Scaling Factor')
plt.ylabel('Hellinger Distance')
plt.legend(fontsize='small', loc='upper right')
plt.grid(True)
plt.savefig("r11_data_visualization.svg")
plt.show()

