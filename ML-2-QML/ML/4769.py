"""Hybrid Classical NAT model that integrates convolutional feature extraction, a lightweight sampler network, and a quantum-inspired filter for enriched feature representation.

The architecture mirrors the original Quantum‑NAT design but replaces the quantum block with a classical analogue that can be paired with the quantum implementation during joint experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """2×2 convolutional filter with a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape (B,1,H,W); extract a 2×2 patch from the center
        pad = self.kernel_size // 2
        patch = F.pad(data, (pad, pad, pad, pad), mode="constant", value=0)
        patch = patch[:, :, :self.kernel_size, :self.kernel_size]
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3], keepdim=True)


class SamplerNetwork(nn.Module):
    """Small MLP that emulates the classical sampler used in Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class QuantumNAT(nn.Module):
    """Classical counterpart of the Quantum‑NAT model."""
    def __init__(self) -> None:
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
        # Classical sampler and filter
        self.sampler = SamplerNetwork()
        self.filter = ConvFilter()
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Normalisation of the concatenated feature vector
        self.norm = nn.BatchNorm1d(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)                       # (B,4)
        # Sampler operates on a pooled representation
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        samp_out = self.sampler(pooled[:, :2])    # (B,2)
        # Classical filter produces a scalar per sample
        filt_out = self.filter(x).squeeze(-1)     # (B,)
        # Concatenate all components
        concatenated = torch.cat([out, samp_out, filt_out.unsqueeze(-1)], dim=1)
        return self.norm(concatenated)


__all__ = ["QuantumNAT"]
