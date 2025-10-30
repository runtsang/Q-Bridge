"""Hybrid self‑attention layer with a Qiskit‑based value extractor.

The class mirrors the classical API but replaces the value projection with a
parameterised quantum circuit that emulates a fully‑connected quantum layer.
The circuit is a simple Ry‑parameterised circuit on n_qubits that produces
an expectation value which is used as the value vector in the attention
mechanism.  The module is fully self‑contained and can be used with any
Aer or real backend.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from typing import Iterable

class HybridSelfAttention:
    """
    Quantum‑augmented self‑attention layer.
    """
    def __init__(self, n_qubits: int = 4, backend: Backend = None, shots: int = 1024):
        """
        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits used to encode each input feature.
        backend : qiskit.providers.Backend, optional
            Quantum backend to execute the circuits.  Defaults to Aer qasm_simulator.
        shots : int, optional
            Number of shots for each execution.
        """
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_value_circuit(self, theta: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that implements a parameterised Ry rotation for each qubit
        and measures all qubits.  The expectation value of the Z‑basis measurement
        is used as the value for the corresponding input feature.
        """
        circuit = QuantumCircuit(self.n_qubits)
        for i, th in enumerate(theta):
            circuit.ry(th, i)
        circuit.measure_all()
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the hybrid attention layer.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters used to compute the query matrix.
        entangle_params : np.ndarray
            Parameters used to compute the key matrix.
        inputs : np.ndarray
            Input batch of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            The attention‑weighted value tensor of shape (batch, embed_dim).
        """
        embed_dim = inputs.shape[1]
        # Classical query and key as in the seed
        query = inputs @ rotation_params.reshape(embed_dim, -1)
        key   = inputs @ entangle_params.reshape(embed_dim, -1)
        scores = np.exp(query @ key.T / np.sqrt(embed_dim))
        scores = scores / scores.sum(axis=-1, keepdims=True)

        # Quantum value extraction
        values = []
        for inp in inputs:
            # Map each feature to a rotation angle; scale to [-π, π]
            theta = 2 * np.pi * (inp - inp.min()) / (inp.max() - inp.min() + 1e-9)
            circuit = self._build_value_circuit(theta)
            job = qiskit.execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            # Convert counts to expectation value of Z for each qubit
            exp = np.zeros(self.n_qubits)
            for state, cnt in counts.items():
                bits = np.array([int(bit) for bit in state[::-1]])
                exp += cnt * (1 - 2 * bits)  # 0 -> +1, 1 -> -1
            exp /= self.shots
            values.append(exp)
        values = np.array(values)  # shape (batch, n_qubits)

        # Weighted sum
        out = scores @ values
        return out

__all__ = ["HybridSelfAttention"]
