""" Runs quantum circuit with various versions of the optimized Fourier pulses. 

The idea is to use a weighted average of the optimized pulse and constant pulse. This prevents issues with the large peak of the optimized pulse
and we can still observe whether it works better than the default. 

Todo: 
 - Write script to analyze result.

"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit import QuantumCircuit
from qiskit import pulse
from qiskit.pulse import Waveform
from qiskit_ibm_provider import IBMProvider

from quantum_gates.utilities import fix_counts, load_config
from pulse_opt.integrals.utilities import load_table_from_pickle
from pulse_opt.pulses.combined_factory import CombinedFactory
from configuration.token import Token


""" Settings """

plt.rcParams['text.usetex'] = False
run = "fourier_extended_constrained"
folder = "data/real"


""" Load available backends. """

# IBMProvider.save_account(token=Token, overwrite=True)

provider = IBMProvider(instance='ibm-q-research-2/uni-trieste-1/main')
backend = provider.get_backend("ibm_nairobi")


""" Load optimized pulses"""
 
config = load_config(f"integrals/{run}.json")
content = config["content"]
df = load_table_from_pickle(run=run)


""" Extract and build optimal one. """

df1 = df.loc[round(df['args.theta'], 2) == 3.14]
data_optimal_pulse = df.iloc[[19]]  # Nr. 19 is optimal
cf = CombinedFactory()
for index, row in data_optimal_pulse.iterrows():
    best_pulse = cf(row)

optimal_waveform = best_pulse.get_pulse()



""" Utilities. """

def compute_Hellinger_distance(p_ng: float, p_real: float, nqubits: int) -> float:
    """ Given two distributions as array, returns the Hellinger distance.
    """
    dh_ng = (np.sqrt(p_real)-np.sqrt(p_ng))**2
    h_ng = 0

    for i in range(2**nqubits):
        h_ng = h_ng + dh_ng[i]

    h_ng = (1/np.sqrt(2)) * np.sqrt(h_ng)
    return h_ng  


def X_circ(depth):
    circ = QuantumCircuit(1,1)
    for i in range(0,depth):
        circ.x(0)
        circ.barrier(range(1))
    circ.measure(range(1),range(1))
    return circ


def main(scaling: float, waveform: callable, shots: int=1000):
    """ Performs run on real hardware of weighted average of optimized pulse and constant pulse. The scaling between 0 and 1 determines 
    the weight of the optimized pulse. Saves the results as txt files.
    """

    # Setup weighted pulse
    print("Setup pulse.")
    constant = lambda x: 1
    weighted_waveform = lambda x: scaling * waveform(x) + (1 - scaling) * constant(x)

    def waveform_scaled(s):
        return weighted_waveform(s/160)

    x = np.linspace(0.0, 160.0, 160)
    y = np.array([np.pi/(4*160) * waveform_scaled(s) for s in x])

    drive_pulse = Waveform(y, name = 'Weighted_Pulse', limit_amplitude = False)

    # Prototype
    qc_list = []
    qubits_layout = [0,1]
    r00_device = []
    r11_device = []

    # Run quantum circuits
    with pulse.build(backend, name='x-gate') as x_q0:
        pulse.play(drive_pulse, pulse.drive_channel(0))

        # Create 10 circuits of different length
        lengths = range(0, 11 * 10, 10)
        for i in lengths:
            qc = X_circ(i)
            qc.add_calibration('x', [0], x_q0)
            qc_list.append(qc)
        
        # Run and wait for result
        job = backend.run(qc_list, shots=shots)
        result = job.result()
        
        # Perform postprocessing
        for i, length in enumerate(lengths):
            counts_0 = result.get_counts(qc_list[i])
            counts = fix_counts(counts_0,1)
            p_real = [counts[j][1]/shots for j in range(0,2)]
            r00_device.append(p_real[0])
            r11_device.append(p_real[1])
            
        # Save results
        np.savetxt(f'{folder}/{run}/r00_X_DEVICE_1_{scaling}.txt', r00_device)
        np.savetxt(f'{folder}/{run}/r11_X_DEVICE_1_{scaling}.txt', r11_device)
        print("Saved results.")
    
    return


if __name__=="__main__": 

    for scaling in np.arange(0.0, 1.0, step=0.1):
        print(f"Start with scaling {scaling}.")
        main(scaling=scaling, waveform=optimal_waveform, shots=1000)
