import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN__gen354(nn.Module):
    """
    Classical sampler network with three hidden layers, batch normalization,
    dropout and ReLU activations.  The architecture is suitable for
    approximating a 2‑dimensional categorical distribution and
    offers better regularisation than the baseline 2‑layer MLP.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 6, 4),
                 output_dim: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution via softmax.
        """
        return F.softmax(self.net(x), dim=-1)

__all__ = ["SamplerQNN__gen354"]
