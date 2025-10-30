"""Quantum latent module for hybrid autoencoder using Qiskit."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def build_latent_circuit(latent_dim: int, reps: int = 5) -> QuantumCircuit:
    """Construct a parameterized circuit that encodes a classical vector and applies a variational ansatz."""
    qc = QuantumCircuit(latent_dim)
    # Input encoding with RX gates
    input_params = [Parameter(f"x_{i}") for i in range(latent_dim)]
    for i, p in enumerate(input_params):
        qc.rx(p, i)
    # Variational ansatz
    ansatz = RealAmplitudes(latent_dim, reps=reps)
    qc.compose(ansatz, inplace=True)
    return qc


class QuantumLatentQNN(SamplerQNN):
    """A quantum neural network that transforms a classical latent vector into a new latent vector."""
    def __init__(
        self,
        circuit_builder: Callable[[int], QuantumCircuit],
        latent_dim: int,
        q_device: str | None = None,
        q_batch_size: int = 1,
    ) -> None:
        circuit = circuit_builder(latent_dim)
        # Separate input and weight parameters
        input_params = [p for p in circuit.parameters if p.name.startswith("x_")]
        weight_params = [p for p in circuit.parameters if p not in input_params]
        sampler = Sampler()
        # Identity interpretation: return raw expectation values
        interpret = lambda x: x
        super().__init__(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            interpret=interpret,
            output_shape=(latent_dim,),
            sampler=sampler,
        )
        self.q_device = q_device
        self.q_batch_size = q_batch_size

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Override to ensure input is a torch.Tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return super().forward(x)


__all__ = ["QuantumLatentQNN", "build_latent_circuit"]
