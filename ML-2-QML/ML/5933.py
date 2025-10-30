import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumKernelSim(nn.Module):
    """Classical simulation of a quantum kernel via a random orthogonal transformation."""
    def __init__(self, in_features: int, out_features: int, seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        # Random orthogonal matrix
        weight = torch.randn(out_features, in_features)
        q, _ = torch.qr(weight)
        self.weight = nn.Parameter(q, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

class HybridQuantumNAT(nn.Module):
    """Hybrid classical network that mimics a quantum kernel using a random orthogonal transform."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 4, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        # Quantum‑kernel simulation
        self.qkernel = QuantumKernelSim(in_features=4*3*3, out_features=4*3*3)
        # Linear head
        self.head = nn.Sequential(
            nn.Linear(4*3*3, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)          # shape (bsz,4,3,3)
        flat = feat.view(bsz, -1)        # (bsz,36)
        kerneled = self.qkernel(flat)    # (bsz,36)
        out = self.head(kerneled)        # (bsz,4)
        out = self.norm(out)
        out = self.dropout(out)
        return out

__all__ = ["HybridQuantumNAT"]
