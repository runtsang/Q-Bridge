"""Hybrid classical model that combines CNN, fully‑connected, sampler, estimator and RBF kernel.

The architecture merges the original Quantum‑NAT CNN with a lightweight
sampler and estimator network, and augments the representation with a
classical radial‑basis kernel that operates on a fixed memory bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """Soft‑max sampler mirroring the QNN helper."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class EstimatorNN(nn.Module):
    """Simple regression network mirroring the EstimatorQNN helper."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class Kernel(nn.Module):
    """Classical RBF kernel used as a memory‑based similarity measure."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridNATModel(nn.Module):
    """Classical hybrid model that combines CNN, fully‑connected, sampler,
    estimator and kernel components."""

    def __init__(self) -> None:
        super().__init__()
        # CNN backbone
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
        self.norm = nn.BatchNorm1d(4)

        # Auxiliary modules
        self.sampler = SamplerModule()
        self.estimator = EstimatorNN()
        self.kernel = Kernel(gamma=1.0)

        # Fixed memory bank for kernel evaluation
        self.memory = torch.randn(10, 4)

    def _kernel_matrix(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix between `x` and the memory bank."""
        bsz = x.shape[0]
        M = memory.shape[0]
        mat = torch.zeros(bsz, M, device=x.device)
        for i in range(bsz):
            for j in range(M):
                mat[i, j] = self.kernel(x[i, :4].unsqueeze(0), memory[j].unsqueeze(0)).item()
        return mat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:  (bsz, 4)
            sample_probs: (bsz, 2)
            regression_output: (bsz, 1)
            kernel_matrix: (bsz, memory_size)
        """
        bsz = x.shape[0]
        # CNN features
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        logits = self.norm(self.fc(flat))

        # Sampler on first two features
        sample_input = flat[:, :2]
        sample_probs = self.sampler(sample_input)

        # Estimator on first feature
        estimator_input = flat[:, :1]
        regression_output = self.estimator(estimator_input)

        # Kernel similarity with memory
        kernel_matrix = self._kernel_matrix(flat, self.memory.to(x.device))

        return logits, sample_probs, regression_output, kernel_matrix


__all__ = ["HybridNATModel"]
