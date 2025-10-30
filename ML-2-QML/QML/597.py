"""Quantum-enhanced quanvolution for MNIST using PennyLane."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple

dev = qml.device("default.qubit", wires=4)

class QuanvolutionFilterQuantum(nn.Module):
    """Variational quantum circuit applied to each 2×2 patch."""
    def __init__(self, n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, 4))

    def _quantum_circuit(self, patch: torch.Tensor, params: torch.Tensor):
        @qml.qnode(dev, interface="torch")
        def circuit(x):
            for i in range(4):
                qml.RY(x[i], wires=i)
            for l in range(self.n_layers):
                for i in range(4):
                    qml.RY(params[l, i], wires=i)
                for i in range(3):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[3, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        return circuit(patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r + 2, c:c + 2].reshape(bsz, 4)
                out = torch.stack([self._quantum_circuit(p, self.params) for p in patch], dim=0)
                patches.append(out)
        features = torch.cat(patches, dim=1)
        return features.view(bsz, -1)

class QuanvolutionClassifierQuantum(nn.Module):
    """Quantum quanvolution followed by a linear classifier."""
    def __init__(self, n_layers: int = 1):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum(n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

def train_quanvolution_quantum(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_layers: int = 1,
) -> Tuple[float, float]:
    """Train the quantum quanvolution on MNIST. Returns train & test accuracy."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuanvolutionClassifierQuantum(n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / total

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    return train_acc, test_acc

__all__ = ["QuanvolutionFilterQuantum", "QuanvolutionClassifierQuantum", "train_quanvolution_quantum"]
