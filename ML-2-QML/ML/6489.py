import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class SamplerQNN_Gen318(nn.Module):
    """
    Classical sampler network that optionally can be extended to a quantum sampler.
    Implements a multi‑layer perceptron with weight‑normalised linear layers,
    ReLU activations and dropout for regularisation.
    """
    def __init__(self,
                 input_dim: int = 2,
                 output_dim: int = 2,
                 hidden_dim: int = 32,
                 hidden_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(weight_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(weight_norm(nn.Linear(in_dim, output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over the output classes.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
