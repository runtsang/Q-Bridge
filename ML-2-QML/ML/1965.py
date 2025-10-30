"""SamplerQNN__gen160 – A robust classical sampler network.

The network is a deeper feed‑forward architecture with batch‑norm, dropout and
custom weight initialization.  It is designed to be plug‑and‑play within
larger pipelines while remaining fully compatible with PyTorch's
autograd engine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN__gen160(nn.Module):
    """
    A multi‑layer neural sampler that outputs a probability distribution over two
    classes.  The architecture is:

    - Linear(2 → 8)
    - BatchNorm1d(8)
    - ReLU
    - Linear(8 → 16)
    - BatchNorm1d(16)
    - ReLU
    - Dropout(0.2)
    - Linear(16 → 2)
    - Softmax

    The weights are initialized with Kaiming normal to improve convergence
    for ReLU activations.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, 2)
            Two‑dimensional feature vector per sample.

        Returns
        -------
        torch.Tensor of shape (batch, 2)
            Probability distribution over two mutually exclusive outcomes.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNN__gen160"]
