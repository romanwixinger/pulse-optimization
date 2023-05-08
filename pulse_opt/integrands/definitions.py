""" Defines the integrand function and makes them available in a lookup.

Attributes:
    integrand_lookup (dict): Lookup with integrand names (str) as keys and the corresponding functions as value.
    integrands (list): List of integrands used in the Ito integrals of the quantum-gates library.
"""

import numpy as np


integrand_lookup = {
    "sin(theta/a)**2": lambda theta, a=1.0: np.sin(theta/a)**2,
    "sin(theta/(2*a))**4": lambda theta, a=1.0: np.sin(theta/(2*a))**4,
    "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a=1.0: np.sin(theta/a)*np.sin(theta/(2*a))**2,
    "sin(theta/(2*a))**2": lambda theta, a=1.0: np.sin(theta/(2*a))**2,
    "cos(theta/a)**2": lambda theta, a=1.0: np.cos(theta/a)**2,
    "sin(theta/a)*cos(theta/a)": lambda theta, a=1.0: np.sin(theta/a)*np.cos(theta/a),
    "sin(theta/a)": lambda theta, a=1.0: np.sin(theta/a),
    "cos(theta/(2*a))**2": lambda theta, a=1.0: np.cos(theta/(2*a))**2,
}
integrands = list(integrand_lookup.keys())
