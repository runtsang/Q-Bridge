"""Hybrid autoencoder combining classical convolution, a quantum latent layer, and a classical decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Quantum neural‑network wrapper
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int = 3, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class QuantumEncoder(nn.Module):
    """Quantum layer that maps a flattened feature vector to a latent space."""
    def __init__(self, circuit: QuantumCircuit, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: torch.tensor(x, dtype=torch.float32),
            output_shape=latent_dim,
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        x_np = x.detach().cpu().numpy()
        outputs = self.qnn.forward(x_np)
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)

class HybridAutoencoderNet(nn.Module):
    """Full hybrid autoencoder with classical conv encoder, quantum latent layer, and classical decoder."""
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 latent_dim: int = 16,
                 hidden_dims: tuple[int,...] = (64, 32),
                 conv_kernel: int = 3,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)

        # Build quantum circuit for latent encoding
        num_qubits = int(np.prod(input_shape))
        self.quantum_circuit = self._build_quantum_circuit(num_qubits, latent_dim)
        self.quantum_encoder = QuantumEncoder(self.quantum_circuit, latent_dim=latent_dim)

        # Decoder: linear layers mirroring encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], np.prod(input_shape)),
            nn.Sigmoid(),
        )
        self.input_shape = input_shape

    def _build_quantum_circuit(self, num_qubits: int, latent_dim: int) -> QuantumCircuit:
        """Construct a simple variational circuit with a RealAmplitudes ansatz."""
        qc = QuantumCircuit(num_qubits)
        qc.append(RealAmplitudes(num_qubits, reps=2), range(num_qubits))
        return qc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)  # (batch, 1, H', W')
        flat = conv_out.view(x.size(0), -1)  # flatten
        latent = self.quantum_encoder(flat)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon_flat = self.decoder(z)
        recon = recon_flat.view(-1, *self.input_shape[1:])
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_shape: tuple[int, int, int],
                latent_dim: int = 16,
                hidden_dims: tuple[int,...] = (64, 32),
                conv_kernel: int = 3,
                conv_threshold: float = 0.0) -> HybridAutoencoderNet:
    """Factory returning a configured hybrid autoencoder."""
    return HybridAutoencoderNet(input_shape, latent_dim, hidden_dims,
                                conv_kernel, conv_threshold)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train only the classical parameters; quantum parameters are frozen."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Freeze quantum encoder
    for param in model.quantum_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoderNet", "Autoencoder", "train_hybrid_autoencoder"]
