""" Defines the integrands in lookups.

Attributes:
    integrands (list): List of integrands used in the Ito integrals of the quantum-gates library.
    equal_weight_lookup (dict): Weight lookup which assigns weight 1.0 to each integrand.
    variance_weight_lookup (dict): Assigns 1.0 to each integrand coming up as variance, rest 0.0.
    covariance_weight_lookup (dict): Assigns 1.0 to each integrand coming up as covariance, rest 0.0.
    deterministic_weight_lookup (dict): Assigns 1.0 to each integrand coming up in the
        deterministic part, rest 0.0.
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

equal_weight_lookup = {key: 1.0 for key in integrand_lookup.keys()}
deterministic_weight_lookup = {
    "sin(theta/a)**2": 1.0,
    "sin(theta/(2*a))**4": 0.0,
    "sin(theta/a)*sin(theta/(2*a))**2": 0.0,
    "sin(theta/(2*a))**2": 0.0,
    "cos(theta/a)**2": 0.0,
    "sin(theta/a)*cos(theta/a)": 0.0,
    "sin(theta/a)": 1.0,
    "cos(theta/(2*a))**2": 1.0,
}
variance_weight_lookup = {
    "sin(theta/a)**2": 1.0,
    "sin(theta/(2*a))**4": 1.0,
    "sin(theta/a)*sin(theta/(2*a))**2": 0.0,
    "sin(theta/(2*a))**2": 0.0,
    "cos(theta/a)**2": 1.0,
    "sin(theta/a)*cos(theta/a)": 0.0,
    "sin(theta/a)": 0.0,
    "cos(theta/(2*a))**2": 0.0,
}
covariance_weight_lookup = {
    "sin(theta/a)**2": 0.0,
    "sin(theta/(2*a))**4": 0.0,
    "sin(theta/a)*sin(theta/(2*a))**2": 1.0,
    "sin(theta/(2*a))**2": 1.0,
    "cos(theta/a)**2": 0.0,
    "sin(theta/a)*cos(theta/a)": 1.0,
    "sin(theta/a)": 1.0,
    "cos(theta/(2*a))**2": 0.0,
}
