"""Hybrid sampler‑classifier (quantum side).

This module implements the quantum component of the hybrid architecture.
It builds a parameterised ansatz inspired by the QuantumClassifierModel
and evaluates it with a Qiskit state‑vector sampler.  The returned
expectation values are fed into a tiny classification head.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Sampler as StatevectorSampler


class HybridSamplerClassifier:
    """
    Quantum implementation of the hybrid sampler‑classifier.

    Parameters
    ----------
    n_qubits : int, default=2
        Number of qubits in the ansatz.
    depth : int, default=1
        Depth of the variational layers.
    backend : str, default='statevector_simulator'
        Backend name used for state‑vector sampling.
    shots : int, default=1024
        Number of shots for qasm simulation; ignored for statevector.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 1,
        backend: str = "statevector_simulator",
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots

        # Parameter vectors
        self.encoding_params = ParameterVector("x", n_qubits)
        self.weight_params = ParameterVector("theta", n_qubits * depth)

        # Build the circuit (mirrors build_classifier_circuit)
        self.circuit = QuantumCircuit(n_qubits)
        for param, qubit in zip(self.encoding_params, range(n_qubits)):
            self.circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(n_qubits):
                self.circuit.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(n_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables for the classification head
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
            for i in range(n_qubits)
        ]

        # Sampler primitive (state‑vector)
        backend_obj = Aer.get_backend(backend)
        self.sampler = StatevectorSampler(backend=backend_obj, shots=shots)

    def run_quantum_expectation(self, encoding: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of encoding parameters.

        Parameters
        ----------
        encoding : np.ndarray
            Shape (batch, n_qubits) of real parameters.

        Returns
        -------
        np.ndarray
            Shape (batch, 1) of expectation values.
        """
        expectations = []
        for row in encoding:
            # Bind encoding parameters
            bind_dict = {self.encoding_params[i]: val for i, val in enumerate(row)}
            bound_circuit = self.circuit.bind_parameters(bind_dict)
            # Get statevector
            sv = Statevector.from_instruction(bound_circuit)
            # Compute average Z expectation over all qubits
            exp = 0.0
            for obs in self.observables:
                exp += sv.expectation_value(obs)
            expectations.append(exp / self.n_qubits)
        return np.array(expectations).reshape(-1, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass mirroring the classical API.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (batch, n_features), where n_features == n_qubits.

        Returns
        -------
        np.ndarray
            Logits of shape (batch, 2).
        """
        # Use input features directly as encoding parameters
        encoding = x.astype(np.float64)
        expectation = self.run_quantum_expectation(encoding)
        # Simple linear head (classical) applied to expectation
        logits = np.hstack([1 - expectation, expectation])  # dummy 2‑class logits
        return logits


def SamplerQNN() -> HybridSamplerClassifier:
    """Compatibility wrapper that returns the quantum hybrid model."""
    return HybridSamplerClassifier()


__all__ = ["HybridSamplerClassifier", "SamplerQNN"]
