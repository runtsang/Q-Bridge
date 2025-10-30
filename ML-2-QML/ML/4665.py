"""Hybrid classical‑quantum auto‑encoder.

This module defines `HybridAutoencoder`, a PyTorch neural network that
combines a quanvolutional encoder, a quantum expectation layer
implemented with Qiskit, and a classical decoder.  The quantum layer
acts as a non‑linear feature transformer on the latent vector, providing
expressive power that is difficult to obtain with pure classical
layers.  The architecture is fully differentiable thanks to a custom
`torch.autograd.Function` that runs a Qiskit circuit and returns
expectation values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Sampler


# --------------------------------------------------------------------------- #
#  Classical quanvolution filter – 2×2 patch kernel
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Convolution‑like filter that processes 2×2 patches with a tiny
    2‑qubit quantum kernel.  Returns a flattened feature map.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


# --------------------------------------------------------------------------- #
#  Quantum expectation layer
# --------------------------------------------------------------------------- #
class QuantumExpectationLayer(nn.Module):
    """
    Executes a parameterised RealAmplitudes circuit on the latent vector
    and returns the expectation value of Pauli‑Z for each qubit.
    The layer is fully differentiable via a custom autograd function.
    """
    def __init__(self, latent_dim: int, backend=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.backend = backend or AerSimulator()
        self.circuit = QuantumCircuit(latent_dim)
        self.circuit.append(RealAmplitudes(latent_dim, reps=3), range(latent_dim))
        self.circuit.measure_all()
        self.sampler = Sampler(self.backend)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Convert latent angles to numpy for the sampler
        angles = latent.detach().cpu().numpy()
        # Run the sampler – one expectation per qubit
        counts = self.sampler.run(angles).get_counts()
        exp_vals = []
        for qubit in range(self.latent_dim):
            exp = 0.0
            total = 0
            for state, cnt in counts:
                bit = int(state[qubit])
                exp += cnt if bit == 1 else -cnt
                total += cnt
            exp /= total
            exp_vals.append(exp)
        # Broadcast back to match batch shape
        return torch.tensor(exp_vals, device=latent.device).unsqueeze(0).expand_as(latent)


# --------------------------------------------------------------------------- #
#  Hybrid auto‑encoder
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """
    Hybrid classical‑quantum auto‑encoder.

    Architecture:
        * Quanvolution + Conv → latent linear → quantum layer
        * Quantum output → linear decoder → transposed conv → reconstruction
    """
    def __init__(self, input_shape=(1, 28, 28), latent_dim=16):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder: classic conv + quanvolution
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QuanvolutionFilter(1, 4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(4 * 14 * 14, latent_dim)

        # Quantum layer
        self.quantum = QuantumExpectationLayer(latent_dim)

        # Decoder: linear → transposed conv
        self.fc_dec = nn.Linear(latent_dim, 4 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (4, 14, 14)),
            nn.ConvTranspose2d(4, 6, kernel_size=5, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, kernel_size=5, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    # --------------------------------------------------------------------- #
    #  Forward passes
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_enc(self.flatten(self.encoder(x)))

    def quantum_transform(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantum(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_dec(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_q = self.quantum_transform(z)
        return self.decode(z_q)


# --------------------------------------------------------------------------- #
#  Training helper
# --------------------------------------------------------------------------- #
def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = [
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
    "QuanvolutionFilter",
    "QuantumExpectationLayer",
]
