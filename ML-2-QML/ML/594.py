import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumHybridClassifier(nn.Module):
    """
    Classical approximation of the hybrid quantum classifier.
    Mirrors the original QCNet architecture but replaces the
    quantum expectation head with a shallow MLP that mimics a
    two‑qubit parameter‑shift circuit.  Batch‑norm and dropout
    layers are added for better generalisation.
    """
    def __init__(self,
                 in_features: int = 55815,
                 hidden: int = 64,
                 n_qubits: int = 2,
                 shift: float = np.pi / 2,
                 dropout: float = 0.1):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(dropout),
            nn.Flatten(),
            nn.Linear(in_features, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, 1)
        )
        # Classical head mimicking a quantum expectation
        self.quantum_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        q_out = self.quantum_head(features)
        return torch.cat((q_out, 1 - q_out), dim=-1)

__all__ = ["QuantumHybridClassifier"]
