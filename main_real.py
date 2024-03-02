""" Top level execution script for testing the optimal pulses with different scaling on real hardware to understand how
the pulse amplitude and form influences the X gate.
"""

import numpy as np

from main.hardware.pulse_scaling import main


if __name__=="__main__":

    run = "fourier_extended_constrained"
    scaling_options = np.arange(0.0, 1.0, step=0.1)
    main(...)
