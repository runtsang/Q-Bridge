import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """Classical sampler producing a 4‑dimensional weight vector."""
    def __init__(self) -> None:
        super().__init__()
        # Map 16‑dim pooled features → 4 weights
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Softmax guarantees weights sum to one, useful for a rotation angle
        return F.softmax(self.net(x), dim=-1)

class QuantumNATHybrid(nn.Module):
    """Hybrid CNN → sampler → quantum → linear → post‑processing."""
    def __init__(self, quantum_circuit) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sampler = SamplerQNN()
        self.quantum = quantum_circuit
        self.fc = nn.Linear(4, 1)          # 4 quantum outputs → 1 target
        self.norm = nn.BatchNorm1d(1)
        # Learnable scale & shift mimic the fraud‑detection style layer
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.size(0)
        feats = self.features(x)
        pooled = F.avg_pool2d(feats, 6).view(bsz, -1)  # shape (bsz, 16)
        weights = self.sampler(pooled)                 # (bsz, 4)
        # Quantum circuit expects a NumPy array; convert per sample
        quantum_out = []
        for w in weights.cpu().numpy():
            quantum_out.append(self.quantum.run(w))
        quantum_out = torch.tensor(np.concatenate(quantum_out, axis=0),
                                   dtype=torch.float32,
                                   device=x.device)
        out = self.fc(quantum_out)
        out = self.scale * out + self.shift
        return self.norm(out)

__all__ = ["SamplerQNN", "QuantumNATHybrid"]
