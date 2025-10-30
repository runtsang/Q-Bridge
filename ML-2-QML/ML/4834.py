"""Hybrid estimator combining classical CNN, fully‑connected layers, and a quantum variational layer.

The architecture mirrors the QCNN and Quantum‑NAT examples:
* Classical feature extractor (conv + pooling) producing 4 latent features.
* Quantum layer that maps each feature vector to a 4‑dimensional expectation vector
  using a small 4‑qubit Ansatz.
* Final linear head to a single output.

The class is fully importable and compatible with the original `EstimatorQNN` API."""
import torch
from torch import nn
import numpy as np
from.EstimatorQNN__gen291_qml import EstimatorQNNGen291Q  # local import

class QuantumLayer(nn.Module):
    """Wraps the Qiskit EstimatorQNN for use inside a PyTorch graph."""
    def __init__(self, qnn: EstimatorQNNGen291Q):
        super().__init__()
        self.qnn = qnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 4)
        x_np = x.detach().cpu().numpy()
        out_np = self.qnn.evaluate(x_np)  # shape (batch, 4)
        return torch.tensor(out_np, device=x.device, dtype=torch.float32)

class EstimatorQNNGen291(nn.Module):
    """Hybrid classical–quantum regressor."""
    def __init__(self):
        super().__init__()
        # Classical convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Map high‑dim feature map to 4 latent features
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Quantum layer producing 4 expectation values
        self.qnn = EstimatorQNNGen291Q()
        self.quantum = QuantumLayer(self.qnn)
        # Final regression head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        latent = self.fc(flat)          # (batch, 4)
        qout = self.quantum(latent)     # (batch, 4)
        return self.head(qout)

__all__ = ["EstimatorQNNGen291"]
