"""Hybrid quantum classifier with self‑attention ansatz and variational classifier.

The class ``HybridClassifier`` builds two independent qiskit circuits:
- a self‑attention style ansatz that encodes the input features into
  rotation and entanglement parameters,
- a depth‑controlled variational classifier that accepts the output of
  the attention block as rotation parameters.

The ``run`` method executes both circuits sequentially on a supplied
backend and returns the classification probabilities derived from the
classifier measurement statistics.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector


class QuantumSelfAttention:
    """Self‑attention style ansatz implemented with Qiskit."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.rx_params = ParameterVector("rx", n_qubits)
        self.ry_params = ParameterVector("ry", n_qubits)
        self.rz_params = ParameterVector("rz", n_qubits)
        self.entangle_params = ParameterVector("entangle", n_qubits - 1)

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(self.rx_params[i], i)
            qc.ry(self.ry_params[i], i)
            qc.rz(self.rz_params[i], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(self.entangle_params[i], i)
        return qc

    def bind_params(
        self, rotation: np.ndarray, entangle: np.ndarray
    ) -> QuantumCircuit:
        """Return a circuit with the parameters bound to the provided values."""
        qc = self.build()
        param_dict = {
            **{p: val for p, val in zip(self.rx_params, rotation)},
            **{p: val for p, val in zip(self.ry_params, rotation)},
            **{p: val for p, val in zip(self.rz_params, rotation)},
            **{p: val for p, val in zip(self.entangle_params, entangle)},
        }
        return qc.bind_parameters(param_dict)


class QuantumClassifierCircuit:
    """Variational classifier ansatz."""

    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        # Variational parameters
        self.weights = ParameterVector("theta", n_qubits * depth)

    def build(self, rotation: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Encode the attention output as rotations
        for i in range(self.n_qubits):
            qc.rx(rotation[i], i)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                qc.ry(self.weights[idx], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
        # Measurement
        qc.measure_all()
        return qc

    def bind_params(self, rotation: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
        qc = self.build(rotation)
        param_dict = {
            **{p: val for p, val in zip(self.weights, weights)},
        }
        return qc.bind_parameters(param_dict)


class HybridClassifier:
    """Hybrid quantum classifier that chains self‑attention and a variational head."""

    def __init__(self, n_qubits: int, depth: int):
        self.attention = QuantumSelfAttention(n_qubits)
        self.classifier = QuantumClassifierCircuit(n_qubits, depth)
        self.backend = Aer.get_backend("qasm_simulator")

    def run(
        self,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the hybrid circuit on the supplied backend.

        Parameters
        ----------
        inputs
            1‑D array of length ``n_qubits`` containing feature values
            in the range ``[0, 2π]``.
        shots
            Number of shots for the simulation.

        Returns
        -------
        probs
            Classification probabilities as a 1‑D array of length 2.
        """
        # Encode input into attention parameters
        rotation = inputs
        entangle = np.zeros(self.attention.n_qubits - 1)

        # Build and execute attention circuit
        attn_qc = self.attention.bind_params(rotation, entangle)
        attn_job = execute(attn_qc, self.backend, shots=shots)
        attn_counts = attn_job.result().get_counts(attn_qc)

        # Convert counts to a normalized probability vector
        attn_vector = np.array(
            [attn_counts.get(bin(i)[2:].zfill(self.attention.n_qubits), 0) for i in range(2 ** self.attention.n_qubits)]
        )
        attn_vector = attn_vector / attn_vector.sum()

        # Use attention output as rotation for classifier
        rotation_cls = attn_vector[: self.classifier.n_qubits]
        weights_cls = np.random.uniform(-np.pi, np.pi, size=self.classifier.weights.size())

        # Build and execute classifier circuit
        cls_qc = self.classifier.bind_params(rotation_cls, weights_cls)
        cls_job = execute(cls_qc, self.backend, shots=shots)
        cls_counts = cls_job.result().get_counts(cls_qc)

        # Map measurement outcomes to binary classification
        probs = np.zeros(2)
        for outcome, count in cls_counts.items():
            if outcome[-1] == "0":  # bit‑0 = class 0
                probs[0] += count
            else:
                probs[1] += count
        probs /= probs.sum()
        return probs


__all__ = ["HybridClassifier"]
