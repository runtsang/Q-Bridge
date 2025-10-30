"""Quantum latent layer for the hybrid autoencoder.

The layer encodes each latent vector as a set of Ry rotations on a
qubit per latent dimension, applies a RealAmplitudes ansatz, and
measures the expectation of Pauli‑Z on each qubit. The output
vector has the same dimensionality as the input latent vector.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class QuantumLatentLayer:
    """Variational quantum layer that transforms latent vectors."""
    def __init__(self, latent_dim: int, reps: int = 2, shots: int = 1024) -> None:
        self.latent_dim = latent_dim
        self.reps = reps
        self.shots = shots

        # Parameters for input rotations and ansatz
        self.input_params = [Parameter(f"theta_{i}") for i in range(latent_dim)]
        self.ansatz_params = [Parameter(f"w_{i}") for i in range(latent_dim * reps)]

        # Circuit template
        self.circuit_template = self._build_circuit()

        # Sampler for expectation values
        self.sampler = Sampler()

        # Quantum parameters as a numpy array for optimisation
        self.weight_params = np.random.rand(len(self.ansatz_params))

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim, "q")
        circuit = QuantumCircuit(qr)
        # Input rotations
        for i, p in enumerate(self.input_params):
            circuit.ry(p, qr[i])
        # Ansatz
        circuit.append(RealAmplitudes(self.latent_dim, reps=self.reps).to_gate(), qr)
        return circuit

    def _bind_circuit(self, params: Iterable[float]) -> QuantumCircuit:
        """Bind the current quantum parameters to the circuit template."""
        mapping = {p: v for p, v in zip(self.ansatz_params, params)}
        return self.circuit_template.assign_parameters(mapping, inplace=False)

    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Transform a batch of latent vectors.

        Args:
            latent: (batch, latent_dim) array of real numbers.
        Returns:
            (batch, latent_dim) array of transformed latent vectors.
        """
        circuit = self._bind_circuit(self.weight_params)
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=self.latent_dim,
            sampler=self.sampler,
        )
        return qnn.forward(latent)

    def set_weight_params(self, params: Iterable[float]) -> None:
        """Update the variational parameters."""
        if len(params)!= len(self.ansatz_params):
            raise ValueError("Parameter length mismatch.")
        self.weight_params = np.array(params, dtype=np.float64)

__all__ = ["QuantumLatentLayer"]
