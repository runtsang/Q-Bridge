"""SamplerQNN: Classical probabilistic sampler with extended features.

This module defines a neural network that maps a 2‑dimensional input to a
probability distribution over two outcomes.  The network is deeper than the
original seed, includes dropout for regularisation, and offers a convenient
`sample` method that draws discrete samples from the output distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """A two‑layer neural sampler with dropout and sampling utilities."""
    def __init__(self,
                 in_features: int = 2,
                 hidden_features: int = 8,
                 out_features: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the two outputs."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self,
               inputs: torch.Tensor,
               n_samples: int = 1) -> np.ndarray:
        """
        Draw discrete samples from the output distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., in_features).
        n_samples : int
            Number of samples to draw per input.

        Returns
        -------
        samples : np.ndarray
            Array of shape (..., n_samples) containing sampled indices {0, 1}.
        """
        probs = self.forward(inputs).detach().cpu().numpy()
        # Broadcast to support batch dimensions
        samples = np.array([np.random.choice(len(p), size=n_samples, p=p)
                            for p in probs])
        return samples

__all__ = ["SamplerQNN"]
