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

# Initialize a figure for plotting
plt.figure()

# Create a plot for each circuit length
for length in circuit_lengths:
    r00_data = []
    r11_data = []
    for scaling in scaling_factors:
        print(f"Load scaling {scaling} for circuit length {length}")
        r00_file = f'{folder}/r00_X_DEVICE_1_{scaling}.txt'
        r11_file = f'{folder}/r11_X_DEVICE_1_{scaling}.txt'
        r00_data.append(load_data(r00_file)[length // 10])  # Assuming data is ordered by circuit length
        r11_data.append(load_data(r11_file)[length // 10])

    # Plotting the Hellinger distance for each scaling factor
    plt.plot(scaling_factors, r00_data, label=f'Circuit Length {length} (r00)')
    plt.plot(scaling_factors, r11_data, label=f'Circuit Length {length} (r11)', linestyle='--')

plt.xlabel('Scaling Factor')
plt.ylabel('Hellinger Distance')
plt.title('Hellinger Distance vs Scaling Factor for Different Circuit Lengths')
plt.legend()
plt.grid(True)
plt.savefig("main_real_visualization.svg")  # Saving the plot with a new filename
plt.show()

