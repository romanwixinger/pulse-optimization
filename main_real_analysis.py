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

# Create a plot for each scaling factor
for scaling in scaling_factors:
    print("Load scaling", scaling)
    r00_data = load_data(f'{folder}/r00_X_DEVICE_1_{scaling}.txt')
    r11_data = load_data(f'{folder}/r11_X_DEVICE_1_{scaling}.txt')

    # Plotting the Hellinger distance for each circuit length
    plt.plot(circuit_lengths, r00_data, label=f'Scaling {scaling:.1f} (r00)')
    plt.plot(circuit_lengths, r11_data, label=f'Scaling {scaling:.1f} (r11)', linestyle='--')

plt.xlabel('Circuit Length')
plt.ylabel('Hellinger Distance')
plt.title('Hellinger Distance vs Circuit Length for Different Scaling Factors')
plt.legend()
plt.grid(True)
plt.savefig("main_real_visualization.svg")



