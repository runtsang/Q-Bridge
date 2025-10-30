"""HybridSamplerQNN: Classical sampler network with fully connected post‑processing.

This module defines a hybrid neural network that combines the SamplerQNN architecture
with a fully connected layer inspired by FCL. It produces probability distributions
and scalar expectations from input tensors or parameter lists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

class HybridSamplerQNN(nn.Module):
    """
    A hybrid sampler network that merges a classical sampler architecture
    with a fully connected layer for expectation computation.
    """
    def __init__(self, n_features: int = 2, n_hidden: int = 4) -> None:
        super().__init__()
        # Sampler architecture
        self.sampler = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2),
        )
        # Fully connected layer to process sampler output
        self.fc = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute sampler probabilities and apply a fully
        connected layer to produce a scalar expectation value.
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        expectation = torch.tanh(self.fc(probs)).mean(dim=-1, keepdim=True)
        return expectation

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the FCL.run method: given a list of thetas, compute a scalar
        expectation value using a linear layer.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.fc(values)).mean(dim=0)
        return expectation.detach().numpy()
