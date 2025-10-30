"""Quantum circuit for fraud detection implemented with PennyLane.

This module defines a variational circuit that mimics the photonic layer
described in the original seed.  The circuit returns the expectation values
of Pauli‑Z on two qubits, which are used as features by the classical model.
"""

import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    """Parameters for the quantum layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudQuantumCircuit:
    """Encapsulates a PennyLane device and a variational circuit."""

    def __init__(self, params: FraudLayerParameters, dev: qml.Device | None = None):
        self.params = params
        self.dev = dev or qml.device("default.qubit", wires=2)

        @qml.qnode(self.dev)
        def circuit():
            # Beam‑splitter equivalent using rotations
            qml.Rot(self.params.bs_theta, self.params.bs_phi, 0.0, wires=0)
            qml.Rot(self.params.bs_theta, self.params.bs_phi, 0.0, wires=1)

            # Phase shifts
            for w, phase in enumerate(self.params.phases):
                qml.PhaseShift(phase, wires=w)

            # Squeezing + displacement approximated by rotations
            for w, (r, phi) in enumerate(zip(self.params.squeeze_r, self.params.squeeze_phi)):
                qml.Rot(r, phi, 0.0, wires=w)

            # Kerr non‑linearity approximated by a controlled‑phase
            for w, k in enumerate(self.params.kerr):
                qml.ControlledPhaseShift(k, wires=[w, (w + 1) % 2])

            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=w)) for w in range(2)]

        self.circuit = circuit

    def __call__(self) -> np.ndarray:
        """Execute the circuit and return a 2‑element numpy array."""
        return np.asarray(self.circuit())
