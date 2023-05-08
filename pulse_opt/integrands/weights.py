""" Defines the weight of each integrand in specific parts of the Noisy gate.

Note:
    Integrands with zero weight are omitted in the definition.

Attributes:
    equal_weight_lookup (dict): Weight lookup which assigns weight 1.0 to each integrand.
    variance_weight_lookup (dict): Assigns 1.0 to each integrand coming up as variance, rest 0.0.
    covariance_weight_lookup (dict): Assigns 1.0 to each integrand coming up as covariance, rest 0.0.
    deterministic_weight_lookup (dict): Assigns 1.0 to each integrand coming up in the
        deterministic part, rest 0.0.
"""


from .definitions import integrand_lookup

equal_weight_lookup = {key: 1.0 for key in integrand_lookup.keys()}
zero_weight_lookup = {key: 0.0 for key in integrand_lookup.keys()}
deterministic_weight_lookup = {
    "sin(theta/a)**2": 1.0,
    "sin(theta/a)": 1.0,
    "cos(theta/(2*a))**2": 1.0,
}
variance_weight_lookup = {
    "sin(theta/a)**2": 1.0,
    "sin(theta/(2*a))**4": 1.0,
    "cos(theta/a)**2": 1.0,
}
covariance_weight_lookup = {
    "sin(theta/a)*sin(theta/(2*a))**2": 1.0,
    "sin(theta/(2*a))**2": 1.0,
    "sin(theta/a)*cos(theta/a)": 1.0,
    "sin(theta/a)": 1.0,
}
variance_plus_deterministic_weight_lookup = {
    "sin(theta/a)**2": 2.0,
    "sin(theta/a)": 1.0,
    "cos(theta/(2*a))**2": 1.0,
    "sin(theta/(2*a))**4": 1.0,
    "cos(theta/a)**2": 1.0,
}

lookup = {
    "equal": equal_weight_lookup,
    "deterministic": deterministic_weight_lookup,
    "variance": variance_weight_lookup,
    "covariance": covariance_weight_lookup,
    "variance_plus_deterministic": variance_plus_deterministic_weight_lookup,
    "zero": zero_weight_lookup,
}
