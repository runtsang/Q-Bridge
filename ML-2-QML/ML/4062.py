import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """
    Hybrid classical model that optionally employs a quantum-inspired filter.
    When `use_quantum=True`, a random linear layer simulates a quantum kernel.
    """

    def __init__(self, use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7
        if self.use_quantum:
            # Simulate a quantum kernel with a linear layer
            self.quantum_kernel = nn.Linear(self.flatten_dim, self.flatten_dim)
        else:
            # Classical 2×2 filter emulating a quantum operation
            self.conv_filter = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        if self.use_quantum:
            qfeat = torch.tanh(self.quantum_kernel(flat))
        else:
            filt = self.conv_filter(x)
            qfeat = filt.view(bsz, -1)
        combined = flat + qfeat
        out = self.fc(combined)
        return self.norm(out)

__all__ = ["QuantumNATHybrid"]
