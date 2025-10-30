"""Hybrid classical module combining a fully connected layer and a sampler network.

The class exposes a forward method that returns both the classical
fully‑connected output and the probability distribution produced by the
sampler network.  The design follows the two seed examples: the
fully‑connected layer mimics the `FCL` seed, while the sampler network
mirrors `SamplerQNN`.  The two sub‑modules are kept independent so
experiments can selectively enable one or both components.

The module is fully compatible with PyTorch and can be dropped into any
training pipeline that expects an nn.Module.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridFCLSampler(nn.Module):
    """Hybrid classical network.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features for the fully connected layer.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Fully connected branch
        self.fcl = nn.Linear(n_features, 1)

        # Sampler branch – 2‑input → 4‑hidden → 2‑output softmax
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, sample_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return outputs of both branches.

        Parameters
        ----------
        x
            Tensor of shape (batch, n_features) for the fully connected layer.
        sample_input
            Tensor of shape (batch, 2) fed to the sampler network.

        Returns
        -------
        fcl_out
            Scalar output from the fully connected branch.
        sampler_out
            2‑dimensional probability distribution from the sampler.
        """
        fcl_out = torch.tanh(self.fcl(x)).mean(dim=0, keepdim=True)
        sampler_out = self.sampler(sample_input)
        return fcl_out, sampler_out

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that mimics the original `run` signature.

        Parameters
        ----------
        thetas
            Tensor of shape (batch, 1) containing parameters for the fully
            connected branch.  The first element of each row is used as
            the input to the linear layer; the remaining elements are
            ignored.

        Returns
        -------
        torch.Tensor
            Expectation value from the fully connected branch.
        """
        # Use only the first column as the input feature
        x = thetas[..., :1]
        return torch.tanh(self.fcl(x)).mean(dim=0)

    def state_dict(self) -> dict:
        """Return a dictionary containing both sub‑module state dicts."""
        return {"fcl": self.fcl.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Load sub‑module state dicts from a dictionary."""
        self.fcl.load_state_dict(state["fcl"])
        self.sampler.load_state_dict(state["sampler"])


__all__ = ["HybridFCLSampler"]
