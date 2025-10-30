"""HybridSamplerQNN: classical autoencoder + quantum sampler."""

from __future__ import annotations

import torch
from torch import nn
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit

# Import the classical autoencoder implementation
from Autoencoder import Autoencoder

class HybridSamplerQNN(nn.Module):
    """
    A hybrid neural network that first encodes input data with a classical
    autoencoder and then samples from a parameterised quantum circuit that
    operates on the latent representation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Classical autoencoder
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Quantum sampler
        self.sampler = Sampler()
        self.input_params = ParameterVector("latent", latent_dim)
        self.qc = self._build_qc(latent_dim)
        self.weight_params = [p for p in self.qc.parameters if p not in self.input_params]
        self.qnn = QSamplerQNN(
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=(2,),
        )

    def _build_qc(self, latent_dim: int) -> QuantumCircuit:
        """Construct a simple RealAmplitudes ansatz that accepts a latent vector."""
        qc = QuantumCircuit(latent_dim)
        qc.compose(RealAmplitudes(latent_dim, reps=3), range(latent_dim), inplace=True)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input and sample from the quantum circuit."""
        with torch.no_grad():
            latent = self.autoencoder.encode(x)
        latent_np = latent.detach().cpu().numpy()
        probs = self.qnn.forward(latent_np)
        return torch.tensor(probs, device=x.device, dtype=torch.float32)

__all__ = ["HybridSamplerQNN"]
