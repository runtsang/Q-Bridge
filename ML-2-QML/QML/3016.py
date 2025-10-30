"""Hybrid autoencoder with quantum encoder and SamplerQNN decoder.

The class implements a quantum autoencoder that encodes classical data into a latent subspace
using a RealAmplitudes ansatz and decodes via a SamplerQNN that reconstructs the input.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42

class HybridAutoencoder:
    def __init__(self, input_dim: int, latent_dim: int = 3, num_trash: int = 2) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=2,
        )
        self.optimizer = COBYLA(maxiter=200)

    def _build_circuit(self) -> None:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encoder ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=5)
        self.circuit.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)

        # Domain wall: flip half of the auxiliary qubits
        for i in range(self.latent_dim + self.num_trash, self.latent_dim + 2 * self.num_trash):
            self.circuit.x(i)

        # Swap test for latent similarity
        aux = self.latent_dim + 2 * self.num_trash
        self.circuit.h(aux)
        for i in range(self.num_trash):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def encode(self, data: np.ndarray) -> torch.Tensor:
        """Encode classical data into a latent vector using the quantum circuit."""
        shots = 1
        result = self.sampler.run(self.circuit, shots=shots, parameter_binds=[{}])
        counts = result.get_counts()
        outcome = int(list(counts.keys())[0])  # 0 or 1
        latent = torch.tensor([outcome], dtype=torch.float32).to("cpu")
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into reconstructed data via SamplerQNN."""
        weights = latents.numpy().flatten()
        sample = self.qnn(weights)
        return torch.tensor(sample, dtype=torch.float32)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        np_data = data.detach().cpu().numpy()
        latents = self.encode(np_data)
        reconstruction = self.decode(latents)
        return reconstruction

    def train(self, data: np.ndarray, epochs: int = 10) -> None:
        """Simple training loop that optimizes the circuit parameters via COBYLA."""
        params = np.array(self.circuit.parameters)
        for epoch in range(epochs):
            def loss_fn(p):
                self.circuit.assign_parameters(p, inplace=True)
                latents = self.encode(data)
                recon = self.decode(latents)
                loss = np.mean((recon - data)**2)
                return loss
            self.optimizer.minimize(loss_fn, params)

    def quantum_decoder(self, latents: torch.Tensor) -> torch.Tensor:
        """Callable decoder that can be attached to the classical hybrid model."""
        return self.decode(latents)

__all__ = ["HybridAutoencoder"]
