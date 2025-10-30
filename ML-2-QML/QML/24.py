"""
Qiskit implementation of a hybrid variational autoencoder using SamplerQNN.
The circuit contains a RealAmplitudes ansatz, a domain‑wall feature injection, and a
swap‑test style measurement.  The class exposes encode, decode, and forward methods
mirroring the classical counterpart, enabling side‑by‑side experiments.
"""

import numpy as np
import torch
from typing import Iterable, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _ensure_rng(seed: int = 42) -> None:
    algorithm_globals.random_seed = seed


class AutoencoderHybrid:
    """Quantum hybrid autoencoder based on a SamplerQNN."""

    def __init__(
        self,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        sampler: StatevectorSampler | None = None,
    ) -> None:
        _ensure_rng()
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = sampler or StatevectorSampler()

        # Build the core autoencoder circuit
        self._build_circuit()

        # Wrap with SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],  # no classical input encoding
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=(2,),
            sampler=self.sampler,
        )

    def _build_circuit(self) -> None:
        """Creates the quantum circuit used for encoding/decoding."""
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Ansatz
        ansatz = RealAmplitudes(num_qubits, reps=self.reps)
        self.circuit.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)
        self.circuit.barrier()

        # Domain‑wall feature injection
        for i in range(self.num_trash):
            self.circuit.x(i + self.latent_dim + self.num_trash)

        # Swap‑test style measurement
        aux = self.latent_dim + 2 * self.num_trash
        self.circuit.h(aux)
        for i in range(self.num_trash):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode classical data into quantum latent representation.
        The inputs are ignored in this simplified example; in practice one would
        use an encoding circuit (e.g., amplitude encoding) before sampling.
        """
        # For demonstration, we treat each sample as a set of parameters for the ansatz
        batch_size = inputs.shape[0]
        params = inputs[:, : len(self.circuit.parameters)].numpy()
        results = self.qnn.forward(params)
        return torch.tensor(results, dtype=torch.float32)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent parameters back into a probability distribution.
        """
        batch_size = latents.shape[0]
        params = latents[:, : len(self.circuit.parameters)].numpy()
        results = self.qnn.forward(params)
        return torch.tensor(results, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        return self.decode(self.encode(inputs))

    def __repr__(self) -> str:
        return (
            f"AutoencoderHybrid(latent_dim={self.latent_dim}, "
            f"num_trash={self.num_trash}, reps={self.reps})"
        )
