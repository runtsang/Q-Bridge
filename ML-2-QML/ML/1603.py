import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNAdvanced(nn.Module):
    """Extended classical sampler network with regularization and deeper layers.
    
    The architecture mirrors the original 2‑to‑2 mapping but adds
    three hidden layers, batch‑norm, dropout and LeakyReLU activations.
    It can be used both as a standalone sampler or as part of a hybrid
    training loop where its parameters are updated jointly with a QNN.
    """
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 8, 4]
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.1))
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities with softmax."""
        return F.softmax(self.net(x), dim=-1)
