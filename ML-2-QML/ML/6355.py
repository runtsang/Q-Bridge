"""
Hybrid CNN + Variational Quantum Layer (QuantumNATEnhanced).

This module defines a PyTorch model that extends the original
QFCModel by adding a variational quantum layer that can be
executed on a simulator or a real quantum device.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VariationalQuantumLayer(nn.Module):
    """
    A wrapper that applies a variational quantum circuit to the input.
    The circuit is defined in the `quantum_circuit` module.
    When `backend` is 'cpu' the layer falls back to a linear transform
    so that the model remains fully classical.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 backend: str = 'default.qubit',
                 device: str = 'cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.device = device

        # Parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        # Fallback linear layer for CPU execution
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum layer.

        Args:
            x: Tensor of shape (batch, in_features)

        Returns:
            Tensor of shape (batch, out_features)
        """
        if self.backend == 'cpu':
            return self.linear(x)

        # Lazy import to keep the module purely classical at import time
        try:
            from quantum_circuit import variational_circuit
        except Exception as exc:
            # If the quantum backend is unavailable we fall back
            # to the classical linear transform
            return self.linear(x)

        return variational_circuit(x, self.params,
                                   backend=self.backend,
                                   device=self.device,
                                   n_qubits=self.n_qubits,
                                   shots=500)

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid CNN + Quantum layer model.

    The architecture re‑uses the convolutional backbone from the original
    QFCModel and appends a variational quantum block that processes the
    flattened feature vector.  The model can be trained end‑to‑end on a
    CPU, GPU or a quantum device by toggling the `run_on_qpu` flag.
    """
    def __init__(self,
                 run_on_qpu: bool = False,
                 qpu_backend: str = 'default.qubit',
                 qpu_device: str = 'cpu'):
        super().__init__()
        self.run_on_qpu = run_on_qpu
        self.qpu_backend = qpu_backend
        self.qpu_device = qpu_device

        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flatten and project to the dimensionality needed for the quantum layer
        self.flatten = nn.Flatten()
        self.fc_proj = nn.Linear(32 * 3 * 3, 16)  # 16 features feed into the quantum block

        # Quantum variational block
        self.quantum_block = VariationalQuantumLayer(
            in_features=16,
            out_features=4,
            n_qubits=4,
            n_layers=3,
            backend=self.qpu_backend,
            device=self.qpu_device
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10-class classification
        )

        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Logits of shape (batch, 10)
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = self.flatten(feat)
        proj = self.fc_proj(flat)

        # Quantum block
        quantum_out = self.quantum_block(proj)

        # Normalise quantum output
        quantum_norm = self.norm(quantum_out)

        logits = self.classifier(quantum_norm)
        return logits

class RandomDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset that generates random images and labels.
    Useful for quick experimentation without external data dependencies.
    """
    def __init__(self,
                 num_samples: int = 1000,
                 img_size: Tuple[int, int] = (28, 28),
                 num_channels: int = 1,
                 num_classes: int = 10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = torch.randn(self.num_channels, *self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label

__all__ = ["QuantumNATEnhanced", "RandomDataset"]
