import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    Classical sampler network that embeds input data with an RBF kernel
    and maps the resulting features to a probability distribution.
    This module can be used to generate weight vectors for a downstream
    quantum sampler, providing a seamless bridge between classical
    feature engineering and quantum state preparation.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        # Two trainable prototype points in the 2‑D input space
        self.prototypes = nn.Parameter(torch.randn(2, 2))
        # Feature extractor → probability generator
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        # Compute RBF kernel features relative to the prototypes
        diff = x[:, None, :] - self.prototypes[None, :, :]
        dist_sq = torch.sum(diff**2, dim=-1)
        kernel_features = torch.exp(-self.gamma * dist_sq)
        # Map features to logits → probabilities
        logits = self.net(kernel_features)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen"]
