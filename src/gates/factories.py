"""
Factories for creating gates with just a function call without any extra arguments. This allows us to give the
arguments upon initialization.

Note:
    The library quantum-gates specifies similar classes with names XFactory, CNOTFactory, etc. These factories
    do not support default arguments.
"""

import numpy as np
from abc import abstractmethod

from quantum_gates.gates import Gates
from quantum_gates.pulses import Pulse


class GateFactory(object):
    """ Makes it possible to specify all arguments for gate construction upon initialization of this class. This saves
        us from passing many unnecessary arguments.
    """

    def __init__(self, pulse: Pulse, gate_args: dict):
        self.pulse = pulse
        self.gate_args = gate_args
        self.gates = Gates(pulse=pulse)

    @abstractmethod
    def construct(self) -> np.array:
        pass


class XGateFactory(GateFactory):

    def __init__(self, pulse: Pulse, gate_args):
        super(XGateFactory, self).__init__(pulse, gate_args)

    def construct(self) -> np.array:
        return self.gates.X(**self.gate_args)


class SXGateFactory(GateFactory):

    def __init__(self, pulse: Pulse, gate_args):
        super(SXGateFactory, self).__init__(pulse, gate_args)

    def construct(self) -> np.array:
        return self.gates.SX(**self.gate_args)


class CRGateFactory(GateFactory):

    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CRGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        return self.gates.CR(**self.gate_args)


class CNOTGateFactory(GateFactory):

    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CNOTGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        return self.gates.CNOT(**self.gate_args)


class CNOTInvGateFactory(GateFactory):

    def __init__(self, pulse: Pulse, gate_args) -> np.array:
        super(CNOTInvGateFactory, self).__init__(pulse, gate_args)

    def construct(self):
        return self.gates.CNOT_inv(**self.gate_args)


factory_class_lookup = {
    "X": XGateFactory,
    "SX": SXGateFactory,
    "CR": CRGateFactory,
    "CNOT": CNOTGateFactory,
    "CNOT_inv": CNOTInvGateFactory,
}
