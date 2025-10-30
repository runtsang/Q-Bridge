"""QuantumNAT_Advanced: Classical backbone with gated residuals and multi-task heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNAT_Advanced(nn.Module):
    """A modernized classical model that leverages quantum‑derived features."""

    def __init__(self, num_tasks: int = 4, hidden_dim: int = 64, pool_type: str = "adaptive"):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Gated residual block
        self.gate = nn.Sequential(
            nn.Linear(16 * 7 * 7, 1),
            nn.Sigmoid()
        )
        self.residual_fc = nn.Linear(16 * 7 * 7, 16 * 7 * 7)
        # Learnable pooling
        if pool_type == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.Identity()
        # Shared fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        # Multi-task heads
        self.task_heads = nn.ModuleDict({
            f"task_{i}": nn.Linear(4, 4) for i in range(num_tasks)
        })
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> dict:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        # Gated residual
        gate_val = self.gate(flattened)
        residual = self.residual_fc(flattened)
        gated = gate_val * residual + (1 - gate_val) * flattened
        # Pooling
        pooled = self.pool(gated.view(bsz, 16, 7, 7)).view(bsz, -1)
        out = self.fc(pooled)
        out = self.norm(out)
        # Multi-task outputs
        outputs = {name: head(out) for name, head in self.task_heads.items()}
        return outputs

__all__ = ["QuantumNAT_Advanced"]
