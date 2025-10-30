"""Hybrid quantum‑classical classifier built with Qiskit.

The class encapsulates a data‑encoding variational circuit and a classical
linear head.  It accepts a NumPy array of shape (n_samples, n_features),
evaluates the circuit on a state‑vector simulator for each sample, extracts
expectation values of Pauli‑Z operators, and feeds them into a linear
classifier.  The interface mirrors the classical ``HybridClassifier`` so
experiments can be run on either side with identical hyper‑parameters.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


class HybridClassifier:
    """Hybrid quantum‑classical classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used for data encoding and variational layers.
    depth : int, default=2
        Number of variational layers.
    latent_dim : int, default=8
        Size of the latent vector produced by the quantum part before
        classification.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        latent_dim: int = 8,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.latent_dim = latent_dim

        # Build the parameterised circuit
        self.circuit, self.encoding_params, self.var_params, self.obs = self._build_circuit()

        # Classical post‑processing head
        self.head = nn.Linear(latent_dim, 2)

    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """Create a data‑encoding circuit with variational layers."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), obs

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Evaluate the circuit on a batch of classical data.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_qubits).

        Returns
        -------
        torch.Tensor
            Logits of shape (n_samples, 2).
        """
        n_samples = x.shape[0]
        latent = []

        # Evaluate each sample on a state‑vector simulator
        for sample in x:
            sv = Statevector.from_label("0" * self.num_qubits)
            bound_circuit = self.circuit.bind_parameters(
                {p: sample[i] for i, p in enumerate(self.encoding_params)}
            )
            sv = sv.evolve(bound_circuit)
            exps = [sv.expectation_value(op).real for op in self.obs]
            latent.append(exps)

        latent = torch.tensor(latent, dtype=torch.float32)
        logits = self.head(latent)
        return logits


__all__ = ["HybridClassifier"]
