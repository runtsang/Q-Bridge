"""Quantum sampler network that implements the same functionality as the classical
SamplerQNNGen270 but using a variational circuit and a Qiskit sampler.

The circuit uses two qubits, receives two classical input parameters, and
includes eight trainable weight parameters that encode both Ry rotations
and a CRX entanglement pattern.  The resulting sampler can be trained
end‑to‑end with PyTorch gradients via Qiskit‑Machine‑Learning.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
import qiskit

class SamplerQNNGen270:
    """
    Quantum sampler network wrapping a variational circuit.

    Architecture
    ------------
    * Two input parameters (2‑D classical data).
    * Eight trainable weight parameters:
        - 4 Ry rotations (two per qubit)
        - 2 additional Ry rotations (self‑attention style)
        - 2 CRX entanglement gates
    * StatevectorSampler for expectation estimation.
    """

    def __init__(self) -> None:
        # Define circuit parameters
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 8)

        # Build variational circuit
        self._circuit = QuantumCircuit(2)
        # Encode inputs
        self._circuit.ry(self.inputs[0], 0)
        self._circuit.ry(self.inputs[1], 1)
        self._circuit.cx(0, 1)

        # Apply weight rotations
        self._circuit.ry(self.weights[0], 0)
        self._circuit.ry(self.weights[1], 1)
        self._circuit.cx(0, 1)
        self._circuit.ry(self.weights[2], 0)
        self._circuit.ry(self.weights[3], 1)

        # Self‑attention‑style rotations
        self._circuit.ry(self.weights[4], 0)
        self._circuit.ry(self.weights[5], 1)

        # Entanglement (CRX) gates
        self._circuit.crx(self.weights[6], 0, 1)
        self._circuit.crx(self.weights[7], 1, 0)

        # Measurement
        self._circuit.measure_all()

        # Sampler primitive
        self.sampler = StatevectorSampler()
        # Wrap into Qiskit‑ML SamplerQNN
        self.sampler_qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def __call__(self, *args, **kwargs):
        """Delegate call to the underlying Qiskit SamplerQNN."""
        return self.sampler_qnn(*args, **kwargs)

    def parameters(self):
        """Return the trainable parameters of the circuit."""
        return self.weights

    def circuit(self):
        """Return the underlying QuantumCircuit."""
        return self._circuit


def SamplerQNN() -> SamplerQNNGen270:
    """Return an instance of the quantum SamplerQNNGen270."""
    return SamplerQNNGen270()


__all__ = ["SamplerQNNGen270", "SamplerQNN"]
