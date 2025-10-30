from __future__ import annotations

import torch
import torch.nn as nn
from.quantum_estimator import EstimatorQNNModel as QuantumEstimator

class QFCModel(nn.Module):
    """Classical CNN followed by a fully‑connected projection to four features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class EstimatorQNNModel(nn.Module):
    """Hybrid estimator that combines classical CNN feature extraction with a variational quantum circuit."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.cnn = QFCModel()
        self.quantum = QuantumEstimator(n_qubits=n_qubits)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.cnn(x)
        quantum_out = self.quantum(features)
        return self.norm(quantum_out)

def EstimatorQNN() -> nn.Module:
    """Return a hybrid estimator combining classical CNN and variational quantum circuit."""
    return EstimatorQNNModel()

__all__ = ["EstimatorQNN", "EstimatorQNNModel"]
