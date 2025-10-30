"""Hybrid classical sampler combining a feed‑forward network with an RBF kernel weighting.

The network produces a probability vector over the two output classes.  The
kernel module evaluates the similarity between the current input and a
pre‑defined set of reference points; the similarity score is used to
re‑weight the network output, providing a data‑driven bias that mimics a
quantum kernel while remaining fully classical.

The class can be dropped into existing pipelines that expect a
`SamplerQNN`‑style interface and retains compatibility with the original
`SamplerQNN` module via an alias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

# --------------------------------------------------------------------------- #
#  Classical RBF kernel implementation
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Encapsulates an RBF kernel with a tunable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return exp(-gamma * ||x-y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper that normalises the input shapes."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# --------------------------------------------------------------------------- #
#  Hybrid sampler network
# --------------------------------------------------------------------------- #
class SamplerQNNGen065(nn.Module):
    """
    A hybrid sampler that combines a small feed‑forward network with an
    RBF‑kernel weighting.  The kernel is evaluated against a fixed set of
    reference points supplied at construction time.
    """

    def __init__(
        self,
        reference_points: Sequence[Sequence[float]],
        gamma: float = 1.0,
        hidden_dim: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        reference_points
            Iterable of 2‑D points that form the kernel support set.
        gamma
            Kernel width hyper‑parameter.
        hidden_dim
            Size of the hidden layer in the neural network.
        """
        super().__init__()
        # Encode reference points as a buffer so they are not treated as a
        # learnable parameter.
        self.register_buffer(
            "ref_points",
            torch.tensor(reference_points, dtype=torch.float32),
        )
        self.kernel = Kernel(gamma)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute a probability distribution over two classes.

        The network output is re‑weighted by the average kernel similarity
        between the input and the reference set.  This mimics the effect of a
        quantum kernel while staying fully classical.
        """
        # Network logits
        logits = self.net(inputs)

        # Kernel similarity to all reference points
        # Shape: (batch, num_refs)
        kernel_vals = self.kernel(inputs.unsqueeze(1), self.ref_points.unsqueeze(0))
        # Average similarity per sample
        weight = kernel_vals.mean(dim=1, keepdim=True)

        # Re‑weight logits
        weighted_logits = logits * weight
        return F.softmax(weighted_logits, dim=-1)

# Alias for backward compatibility with the original module name
SamplerQNN = SamplerQNNGen065

__all__ = ["SamplerQNNGen065", "SamplerQNN"]
