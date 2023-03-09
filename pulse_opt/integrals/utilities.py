"""Utilities used for characterizing the integrals.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:
    integrands (list): List of integrands used in the Ito integrals of the quantum-gates library.

    markers (list): List of matplotlib.pyplot markers used for visualizing the integration values.
"""


integrands = [
    "sin(theta/a)**2",
    "sin(theta/(2*a))**4",
    "sin(theta/a)*sin(theta/(2*a))**2",
    "sin(theta/(2*a))**2",
    "cos(theta/a)**2",
    "sin(theta/a)*cos(theta/a)",
    "sin(theta/a)",
    "cos(theta/(2*a))**2"
]

markers = [".", "^", "o", "2", "*", "D", "x", "X", "+"]
