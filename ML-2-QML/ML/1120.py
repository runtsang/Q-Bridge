"""SamplerQNN: a lightweight neural sampler with dropout and batch normalization.

This module extends the original two‑layer MLP by adding dropout layers for regularization
and batch‑normalization to stabilize training. It also exposes a convenient
`sample` method that draws discrete samples from the output probability
distribution, making it ready for downstream generative tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """
    A two‑layer neural sampler with dropout and batch normalization.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dim : int, default 8
        Number of hidden units.
    output_dim : int, default 2
        Number of classes / output probabilities.
    dropout : float, default 0.3
        Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch_size, output_dim).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> np.ndarray:
        """
        Draw discrete samples from the predicted distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        num_samples : int, default 1
            Number of samples to draw per input instance.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size, num_samples) containing sampled class indices.
        """
        probs = self.forward(inputs).detach().cpu().numpy()
        return np.array(
            [np.random.choice(len(p), size=num_samples, p=p) for p in probs]
        )

__all__ = ["SamplerQNN"]
