from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Tuple

class HybridAutoencoder(qml.nn.Module):
    def __init__(self,
                 num_qubits: int = 4,
                 latent_dim: int = 4,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        # Classical convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Map encoder output to quantum circuit parameters
        self.param_mapper = nn.Linear(16 * 7 * 7, num_qubits)
        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # Quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def qnode(params: torch.Tensor) -> torch.Tensor:
            qml.templates.AngleEmbedding(params, wires=range(num_qubits))
            # Simple ansatz: one layer of RealAmplitudes
            qml.templates.RealAmplitudes(self.weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        self.weights = nn.Parameter(0.01 * torch.randn(1, num_qubits))
        self.qnode = qnode
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(num_qubits, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], 16 * 7 * 7),
            nn.ReLU(),
        )
        self.reconstruction = nn.Sequential(
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        params = self.param_mapper(flattened)
        return params

    def quantum(self, params: torch.Tensor) -> torch.Tensor:
        return self.qnode(params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(z)
        return self.reconstruction(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.encode(x)
        z = self.quantum(params)
        return self.decode(z)

__all__ = ["HybridAutoencoder"]
