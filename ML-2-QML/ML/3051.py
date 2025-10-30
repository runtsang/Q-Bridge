import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableQuantumHead(nn.Module):
    """A lightweight head that emulates a quantum expectation using a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        probs = torch.sigmoid(logits + self.shift)
        return probs

class EstimatorRegression(nn.Module):
    """Feed‑forward regression network inspired by Qiskit EstimatorQNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridQCNet(nn.Module):
    """CNN backbone followed by a shared trunk and two heads."""
    def __init__(self, use_quantum_head: bool = False):
        super().__init__()
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Shared fully‑connected trunk
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        # Heads
        self.cls_head = DifferentiableQuantumHead(84)
        self.reg_head = EstimatorRegression()
        self.use_quantum_head = use_quantum_head

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.shared_fc(x)
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        return {"probability": cls, "regression": reg}

__all__ = ["DifferentiableQuantumHead", "EstimatorRegression", "HybridQCNet"]
