import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """Classical surrogate for a quantum fully‑connected layer."""
    def __init__(self, n_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

class QuantumNATHybrid(nn.Module):
    """Hybrid classical model that mimics the Quantum‑NAT architecture and adds an
    FCL‑style expectation layer."""
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
            nn.Linear(64, 4)
        )
        self.fcl = FullyConnectedLayer(n_features=4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        fc_out = self.fc(flattened)
        # Convert tensor to list of floats for the classical FCL simulation
        thetas = fc_out.detach().cpu().numpy().tolist()
        expectation = self.fcl.run(thetas)
        out = torch.tensor(expectation, device=x.device, dtype=torch.float32).view(bsz, -1)
        return self.norm(out)

__all__ = ["QuantumNATHybrid"]
