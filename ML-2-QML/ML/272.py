"""Enhanced classical sampler network with dropout and sampling utilities.

The SamplerQNN class mirrors the original interface but adds a richer
architecture: two hidden layers, dropout for regularisation, and a
``sample`` method that draws from the output probability distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class SamplerQNN(nn.Module):
    """
    A deeper neural sampler with dropout regularisation.

    Parameters
    ----------
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch, 2)``.
        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape ``(batch, 2)``.
        """
        return F.softmax(self.net(inputs), dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch, 2)``.
        num_samples : int
            Number of samples per input.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(batch, num_samples)`` with values 0 or 1.
        """
        probs = self.forward(inputs)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

__all__ = ["SamplerQNN"]
