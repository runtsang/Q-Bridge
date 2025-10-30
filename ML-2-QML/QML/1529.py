"""Advanced variational sampler circuit with multi‑qubit entanglement."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as _QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as _StatevectorSampler


class AdvancedSamplerQNN:
    """
    A parameterised quantum circuit designed for sampling tasks.
    The circuit operates on four qubits, uses entangling blocks,
    and is wrapped by Qiskit Machine Learning's SamplerQNN for easy
    integration into classical optimisation loops.

    The class exposes the underlying `QuantumCircuit` and the
    `SamplerQNN` wrapper, providing a convenient interface for
    training and evaluation.
    """

    def __init__(self, n_qubits: int = 4, entangle: bool = True) -> None:
        self.n_qubits = n_qubits
        self.entangle = entangle

        # Define input and weight parameters
        self.input_params = ParameterVector("x", self.n_qubits)
        self.weight_params = ParameterVector("theta", 8)

        # Build the circuit
        self.circuit = self._build_circuit()

        # Wrap with Qiskit ML sampler
        self.sampler = _StatevectorSampler()
        self.sampler_qnn = _QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a deep, entangled variational circuit."""
        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr)

        # Initial rotation layer
        for i, param in enumerate(self.input_params):
            qc.ry(param, i)

        # Two entangling blocks with parameterised rotations
        for block in range(2):
            # Entangle adjacent qubits
            if self.entangle:
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                    qc.cx(i + 1, i)
            # Parameterised rotations
            for i, param in enumerate(self.weight_params[block * 4 : (block + 1) * 4]):
                qc.ry(param, i % self.n_qubits)

        # Final measurement is implicit; sampler will return statevector probabilities
        return qc

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_sampler_qnn(self) -> _QiskitSamplerQNN:
        """Return the wrapped Qiskit SamplerQNN instance."""
        return self.sampler_qnn


__all__ = ["AdvancedSamplerQNN"]
