"""Hybrid quantum autoencoder.

The implementation follows the structure of the classical model: an encoder
circuit that maps classical features into a quantum state, a variational
ansatz that compresses the state into a latent sub‑space, and a decoder that
reconstructs the original features.  The circuit is built using Qiskit's
SamplerQNN, which provides a differentiable interface that can be embedded
in a PyTorch training loop.  A domain‑wall sub‑circuit is inserted to
demonstrate how logical operations can be combined with the variational
layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, X, H
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.primitives import Sampler
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

def domain_wall_circuit(num_qubits: int, start: int = 0, end: int | None = None) -> QuantumCircuit:
    """Injects a domain wall (X gates) into a contiguous block of qubits."""
    end = end or num_qubits
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc

def build_autoencoder_circuit(
    num_features: int,
    latent_dim: int,
    num_trash: int = 2,
) -> QuantumCircuit:
    """
    Construct a quantum circuit that implements an auto‑encoder.

    1. Encode the classical feature vector using RawFeatureVector.
    2. Apply a RealAmplitudes ansatz on the first `latent_dim` qubits
       (the latent sub‑space).
    3. Perform a swap‑test between the latent sub‑space and a set of
       trash qubits to enforce fidelity between the encoded and decoded
       states.
    4. Apply a domain‑wall sub‑circuit to demonstrate logical gate
       composition.
    """
    total_qubits = num_features + num_trash
    qc = QuantumCircuit(total_qubits)

    # 1. Feature encoding
    feature_map = RawFeatureVector(num_qubits=num_features, reps=1)
    qc.compose(feature_map, range(num_features), inplace=True)

    # 2. Variational ansatz on latent sub‑space
    ansatz = RealAmplitudes(latent_dim, reps=3)
    qc.compose(ansatz, range(latent_dim), inplace=True)

    # 3. Swap test with trash qubits
    aux = latent_dim + num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, latent_dim + i, i)
    qc.h(aux)

    # 4. Domain wall on remaining qubits
    qc.compose(domain_wall_circuit(total_qubits, start=latent_dim + num_trash), range(latent_dim + num_trash, total_qubits), inplace=True)

    qc.measure_all()
    return qc

class QuantumAutoencoder(nn.Module):
    """
    PyTorch wrapper around a SamplerQNN that implements a quantum auto‑encoder.
    """
    def __init__(
        self,
        num_features: int,
        latent_dim: int,
        shots: int = 1024,
        backend: AerSimulator | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.backend = backend or AerSimulator()
        self.sampler = Sampler(method="automatic", backend=self.backend)
        circuit = build_autoencoder_circuit(num_features, latent_dim)
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=(num_features,),
            sampler=self.sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects an input tensor of shape (batch, num_features).
        The output is a reconstruction of the same shape.
        """
        flat_inputs = inputs.reshape(-1, self.num_features)
        recon = self.qnn(flat_inputs)
        return recon.reshape(inputs.shape)

__all__ = ["QuantumAutoencoder", "build_autoencoder_circuit", "domain_wall_circuit"]
