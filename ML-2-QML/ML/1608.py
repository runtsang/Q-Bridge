import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianCalibration(nn.Module):
    """Calibrates logits using a prior‑weighted sigmoid."""
    def __init__(self, prior: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.prior = prior
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((logits - self.prior) / (1 + self.prior))

class HybridLayer(nn.Module):
    """Hybrid layer that maps a dense vector to multi‑qubit probabilities."""
    def __init__(self, in_features: int, n_qubits: int = 4, shift: float = 0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.linear = nn.Linear(in_features, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.linear(x)
        expectations = torch.cos(angles)
        probs = (expectations.sum(dim=-1, keepdim=True) + self.n_qubits) / (2 * self.n_qubits)
        return probs

class QuantumHybridBinaryClassifier(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model."""
    def __init__(self, n_qubits: int = 4, shift: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = HybridLayer(self.fc3.out_features, n_qubits, shift)
        self.calib = BayesianCalibration()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        probs = self.calib(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["BayesianCalibration", "HybridLayer", "QuantumHybridBinaryClassifier"]
