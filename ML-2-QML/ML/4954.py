"""Hybrid classical model combining CNN, regression, sampler, and estimator sub‑networks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def SamplerQNN() -> nn.Module:
    """Simple feed‑forward sampler network mirroring the quantum SamplerQNN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(x), dim=-1)

    return SamplerModule()


def EstimatorQNN() -> nn.Module:
    """Simple regression network mirroring the quantum EstimatorQNN."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
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

    return EstimatorNN()


class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid model that processes image‑like inputs through a CNN,
    projects to a regression head, and optionally feeds the latent vector into
    sampler and estimator networks that emulate quantum sub‑modules.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor (as in QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Optional quantum‑style sub‑networks
        self.sampler = SamplerQNN()
        self.estimator = EstimatorQNN()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns a dictionary with keys:
            'features' : CNN feature map
           'regression' : normalized regression output
           'sampler' : sampler network output
            'estimator' : estimator network output
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        regression = self.norm(self.regressor(flat))

        # Treat the regression output as a 2‑dimensional input for sampler/estimator
        # (use only first two dimensions to avoid shape mismatch)
        sampler_in = regression[:, :2]
        estimator_in = regression[:, :2]

        sampler_out = self.sampler(sampler_in)
        estimator_out = self.estimator(estimator_in)

        return {
            "features": feats,
            "regression": regression,
            "sampler": sampler_out,
            "estimator": estimator_out,
        }


__all__ = ["HybridQuantumNAT"]
