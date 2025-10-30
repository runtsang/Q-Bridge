import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class SamplerQNN(nn.Module):
    """
    Classical sampler network with configurable architecture.
    Supports optional BatchNorm and Dropout for regularisation.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (4, 4),
        output_dim: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Size of the input feature vector.
        hidden_dims : Sequence[int]
            Sizes of hidden layers.
        output_dim : int
            Size of the output probability vector.
        dropout : float
            Dropout probability applied after each hidden layer.
        use_batchnorm : bool
            Whether to insert BatchNorm layers after linear transforms.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over output_dim classes.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample discrete outcomes according to the output distribution.
        Returns a tensor of shape (batch_size, num_samples).
        """
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).permute(1, 0)

    def load_pretrained(self, state_dict: dict) -> None:
        """
        Load pretrained weights into the network.
        """
        self.load_state_dict(state_dict)
