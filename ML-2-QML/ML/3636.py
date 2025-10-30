"""Hybrid quantum‑classical neural network.

The module defines a classical neural network that mirrors the SamplerQNN
architecture but adds a fully‑connected read‑out.  It can be trained
with standard deep‑learning pipelines and optionally accepts external
weight vectors for inference.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def HybridQNN():
    """Return a classical neural network that combines a sampler network
    with a fully‑connected layer.
    """
    class HybridModule(nn.Module):
        def __init__(self, n_features: int = 2) -> None:
            super().__init__()
            # Sampler‑style feature extractor
            self.extractor = nn.Sequential(
                nn.Linear(n_features, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
                nn.Softmax(dim=-1),
            )
            # Fully‑connected read‑out
            self.fc = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass returning a scalar prediction."""
            probs = self.extractor(x)
            out = self.fc(probs)
            return out

        def run(self, thetas: Iterable[float], dummy_input: torch.Tensor | None = None) -> np.ndarray:
            """
            Optionally inject external weight values for inference.
            ``thetas`` must contain exactly the number of parameters in
            the network (6 in total: 4 for the extractor, 2 for the fc).
            """
            # Load external weights if provided
            if thetas is not None:
                idx = 0
                for param in self.parameters():
                    shape = param.shape
                    n = torch.numel(param)
                    param.data.copy_(torch.tensor(thetas[idx:idx + n], dtype=param.dtype).view(shape))
                    idx += n

            # Default input: zero vector of matching dimensionality
            if dummy_input is None:
                dummy_input = torch.zeros(1, self.extractor[0].in_features)

            with torch.no_grad():
                output = self.forward(dummy_input)
            return output.detach().cpu().numpy()

    return HybridModule()
