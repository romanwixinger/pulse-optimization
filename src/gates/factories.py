"""Factories for creating gates with just a function call without any extra arguments.

This allows us to give the arguments upon initialization. Note that The library quantum-gates specifies similar classes with names XFactory, CNOTFactory, etc. These factories
    do not support default arguments.

Attributes:
    factory_class_lookup (dict): Lookup with the gate name (str) as key and the corresponding factory class
        (GateFactory) as value.
"""

import numpy as np
from abc import abstractmethod

from quantum_gates.gates import Gates
from quantum_gates.pulses import Pulse


class GateFactory(object):
    """Creates gates with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """

    def __init__(self, pulse: Pulse, gate_args: dict):
        self.pulse = pulse
        self.gate_args = gate_args
        self.gates = Gates(pulse=pulse)

    @abstractmethod
    def construct(self) -> np.array:
        """Samples a gate as specified by the attributes.

            Abstract method which is overwritten in the child classes to sample a specific gate.

            Returns:
                np.array: The sampled gate.
        """
        pass


class XGateFactory(GateFactory):
    """Creates an X gate with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """

    def __init__(self, pulse: Pulse, gate_args):
        super(XGateFactory, self).__init__(pulse, gate_args)

    def construct(self) -> np.array:
        """Samples an X gate as specified by the attributes.

            Returns:
                np.array: The sampled gate.
        """
        return self.gates.X(**self.gate_args)


class SXGateFactory(GateFactory):
    """Creates an SX gate with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """
    def __init__(self, pulse: Pulse, gate_args):
        super(SXGateFactory, self).__init__(pulse, gate_args)

    def construct(self) -> np.array:
        """Samples a SX gate as specified by the attributes.

            Returns:
                np.array: The sampled gate.
        """
        return self.gates.SX(**self.gate_args)


class CRGateFactory(GateFactory):
    """Creates a CR gate with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """
    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CRGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        """Samples a CR gate as specified by the attributes.

            Returns:
                np.array: The sampled gate.
        """
        return self.gates.CR(**self.gate_args)


class CNOTGateFactory(GateFactory):
    """Creates a CNOT gate with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """
    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CNOTGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        """Samples a CNOT gate as specified by the attributes.

        Returns:
            np.array: The sampled gate.
        """
        return self.gates.CNOT(**self.gate_args)


class CNOTInvGateFactory(GateFactory):
    """Creates an CNOT inverse gate with a argument-free function call.

        Attributes:
            pulse (Pulse): Pulse used in the gates (Gates).
            gate_args (dict): Arguments that are used to sample the gates. These are the inputs of the construct()
                method in the Gates class.
    """
    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CNOTInvGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        """Samples a CNOT inverse gate as specified by the attributes.

            Returns:
                np.array: The sampled gate.
        """
        return self.gates.CNOT_inv(**self.gate_args)


factory_class_lookup = {
    "X": XGateFactory,
    "SX": SXGateFactory,
    "CR": CRGateFactory,
    "CNOT": CNOTGateFactory,
    "CNOT_inv": CNOTInvGateFactory,
}
