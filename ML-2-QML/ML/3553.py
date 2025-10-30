"""Hybrid sampler‑estimator neural network.

This class mirrors the classical SamplerQNN and extends it with a regression head
inspired by EstimatorQNN. The architecture shares a backbone and splits into
two heads: a softmax sampler and a linear estimator. It facilitates joint
training of classification and regression objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNN(nn.Module):
    """
    Two‑head neural network producing a sampling probability distribution
    and a regression prediction.

    Architecture
    ------------
    - Shared backbone: Linear(2 → 8) → Tanh → Linear(8 → 4) → Tanh
    - Sampler head: Linear(4 → 2) → Softmax
    - Estimator head: Linear(4 → 1)

    The network can be trained with the sum of cross‑entropy loss (for the
    sampler) and mean‑squared error (for the estimator).
    """
    def __init__(self) -> None:
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        # Sampler head
        self.sampler_head = nn.Linear(4, 2)
        # Estimator head
        self.estimator_head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of inputs of shape (batch_size, 2).

        Returns
        -------
        probs : torch.Tensor
            Probabilities of shape (batch_size, 2) after softmax.
        preds : torch.Tensor
            Regression predictions of shape (batch_size, 1).
        """
        shared_out = self.shared(inputs)
        probs = F.softmax(self.sampler_head(shared_out), dim=-1)
        preds = self.estimator_head(shared_out)
        return probs, preds

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Sample from the learned probability distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.
        device : torch.device or str, default "cpu"
            Device on which to perform the sampling.

        Returns
        -------
        samples : torch.Tensor
            Integer tensor of shape (batch_size,) with values 0 or 1.
        """
        with torch.no_grad():
            dummy = torch.zeros(batch_size, 2, device=device)
            probs, _ = self.forward(dummy)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def state_dict(self, *args, **kwargs):
        """Override to include estimator head parameters."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Override for consistency."""
        return super().load_state_dict(*args, **kwargs)

__all__ = ["SamplerQNN"]
