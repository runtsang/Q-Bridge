"""Hybrid classical CNN with optional quantum skip connection.

The model can operate in three modes:
  * classical-only (default)
  * quantum-only (if `use_quantum=True` and a quantum block is supplied)
  * hybrid (both paths combined via a learnable weight).

The design enables side‑by‑side experiments and ablation studies
without modifying the forward signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATModel(nn.Module):
    """Classical backbone with a learnable skip to an optional quantum block."""

    def __init__(self, use_quantum: bool = False, quantum_block: nn.Module | None = None):
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_block = quantum_block

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Learnable fusion weight (0–1)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # Normalisation
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantum fusion."""
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        classical = self.fc(flat)

        if self.use_quantum and self.quantum_block is not None:
            # Quantum block must accept a tensor and return a tensor of shape (bsz, 4)
            quantum = self.quantum_block(x)
            # Fuse with learnable weight
            out = self.fusion_weight * classical + (1.0 - self.fusion_weight) * quantum
        else:
            out = classical

        return self.norm(out)

__all__ = ["QuantumNATModel"]
