"""Advanced classical sampler network.

Provides a configurable multi‑layer MLP with dropout, custom weight
initialisation and a simple sampling API that mirrors the quantum
implementation.  The class can be used both for supervised learning
and for hybrid quantum‑classical experiments."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSamplerQNN(nn.Module):
    """
    A multi‑layer perceptron that maps a 2‑dimensional input to a
    probability distribution over two classes.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (default 2).
    hidden_dims : list[int]
        Sequence of hidden layer sizes.  A longer list yields a deeper
        network (default [8, 8]).
    output_dim : int
        Number of output logits (default 2).
    dropout : float
        Dropout probability applied after every hidden layer
        (default 0.2).
    init_std : float
        Standard deviation for normal weight initialization
        (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        dropout: float = 0.2,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 8]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over outputs."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor | None = None, n_samples: int = 1000) -> torch.Tensor:
        """
        Generate samples from the categorical distribution produced by
        the network.

        Parameters
        ----------
        inputs : torch.Tensor, optional
            Input tensor of shape (batch, input_dim).  If None, a
            uniform batch of size ``n_samples`` is used.
        n_samples : int
            Number of samples to return when ``inputs`` is None.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_samples, output_dim) containing
            one‑hot encoded samples.
        """
        if inputs is None:
            inputs = torch.randn(n_samples, self.net[0].in_features)
        probs = self.forward(inputs)
        cat = torch.distributions.Categorical(probs)
        samples = cat.sample((n_samples,))
        return F.one_hot(samples, num_classes=probs.shape[-1]).float()


__all__ = ["AdvancedSamplerQNN"]
