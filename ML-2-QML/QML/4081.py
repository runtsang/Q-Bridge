"""Hybrid quantum‑classical autoencoder.

The `HybridAutoencoder` class implements the same interface as the
classical version but replaces the latent bottleneck with a Qiskit
`SamplerQNN`.  The quantum circuit embeds the latent vector via
parameterised Ry rotations, applies a RealAmplitudes ansatz, and
measures each qubit to produce a vector of expectation values.  A
domain‑wall X gate is applied to the first qubit to emulate the
domain‑wall idea from the original quantum seed.  Parameter clipping
follows the FraudDetection style.

Usage
-----
    model = HybridAutoencoder(config)
    reconstruction = model(data)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RealAmplitudes
import torch
from torch import nn
from typing import Tuple
from dataclasses import dataclass


class ClippedLinear(nn.Module):
    """Linear layer that clips its parameters after each forward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.clip_low, self.clip_high = clip_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.linear.weight.clamp_(self.clip_low, self.clip_high)
            if self.linear.bias is not None:
                self.linear.bias.clamp_(self.clip_low, self.clip_high)
        return self.linear(x)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    quantum_qubits: int | None = None  # If None, use classical latent


class HybridAutoencoder(nn.Module):
    """Hybrid quantum‑classical autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(
                ClippedLinear(in_dim, hidden, clip_range=config.clip_range)
            )
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(
            ClippedLinear(in_dim, config.latent_dim, clip_range=config.clip_range)
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum latent layer
        if config.quantum_qubits is None or config.quantum_qubits <= 0:
            # Purely classical latent
            self.quantum_layer = nn.Identity()
        else:
            num_qubits = config.quantum_qubits
            latent_dim = config.latent_dim
            # Input parameters for latent vector encoding
            input_params = [Parameter(f"theta_{i}") for i in range(latent_dim)]
            # Basic circuit
            qc = QuantumCircuit(num_qubits)
            for i, param in enumerate(input_params):
                qc.ry(param, i)  # Angle encoding
            # Domain‑wall: flip first qubit
            qc.x(0)
            # Ansatz
            ansatz = RealAmplitudes(num_qubits, reps=3)
            qc.append(ansatz, range(num_qubits))
            # Measure all qubits
            qc.measure_all()
            # Sampler
            sampler = StatevectorSampler()
            self.quantum_layer = SamplerQNN(
                circuit=qc,
                input_params=input_params,
                weight_params=ansatz.parameters,
                interpret=lambda x: x,
                output_shape=latent_dim,
                sampler=sampler,
            )

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(
                ClippedLinear(in_dim, hidden, clip_range=config.clip_range)
            )
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(
            ClippedLinear(in_dim, config.input_dim, clip_range=config.clip_range)
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        # If quantum layer is identity, skip conversion.
        if isinstance(self.quantum_layer, nn.Identity):
            return self.decoder(z)
        # Quantum layer expects a NumPy array.
        z_np = z.detach().cpu().numpy()
        qz_np = self.quantum_layer(z_np)
        qz = torch.from_numpy(qz_np).to(z.device)
        return self.decoder(qz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        return self.decode(self.encode(x))


__all__ = ["AutoencoderConfig", "HybridAutoencoder"]
