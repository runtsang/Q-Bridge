import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np

class SamplerQNN(nn.Module):
    """
    A flexible classical sampler network.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_dims : Sequence[int]
        Sizes of hidden layers. Defaults to a single 4‑unit hidden layer.
    output_dim : int
        Number of output classes. Defaults to 2 for binary sampling.
    dropout : float
        Dropout probability applied after each hidden layer.
    init_std : float
        Standard deviation for normal weight initialization.

    Notes
    -----
    The network ends with a softmax to produce a probability distribution
    over the output classes.  The :meth:`sample` method draws discrete
    samples from this distribution.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (4,),
        output_dim: int = 2,
        dropout: float = 0.0,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Custom weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Probability distribution of shape (batch, output_dim).
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, probs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the output probability distribution.

        Parameters
        ----------
        probs : torch.Tensor
            Tensor of probabilities (batch, output_dim).
        n_samples : int
            Number of samples to draw per batch element.

        Returns
        -------
        torch.Tensor
            Integer samples of shape (batch, n_samples).
        """
        batch_size, num_classes = probs.shape
        probs_np = probs.detach().cpu().numpy()
        samples = np.random.choice(
            a=num_classes,
            size=(batch_size, n_samples),
            p=probs_np
        )
        return torch.tensor(samples, dtype=torch.long, device=probs.device)

__all__ = ["SamplerQNN"]
