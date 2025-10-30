from __future__ import annotations

import torch
from torch import nn

class EstimatorQNNGen227(nn.Module):
    """Hybrid‑friendly feed‑forward network mirroring the quantum estimator interface.

    The network is parameterised by ``depth`` which determines the number of
    linear layers.  ``encoding`` returns the indices of the inputs that
    correspond to the quantum encoding variables.  ``weight_sizes`` is a list
    of the number of trainable parameters per layer, matching the quantum
    weight vector length.  ``observables`` are dummy placeholders that can be
    used to construct a quantum observable list in the QML counterpart.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        # Build depth many hidden layers
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Metadata for quantum mapping
        self.encoding = list(range(input_dim))
        # Weight sizes per layer (weights + bias)
        self.weight_sizes = [p.numel() for p in self.parameters()]
        # Observables placeholder: one per output neuron
        self.observables = list(range(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weight_vector(self) -> torch.Tensor:
        """Flatten all parameters into a 1‑D vector matching the quantum weight vector."""
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_weight_vector(self, vec: torch.Tensor) -> None:
        """Set the network weights from a 1‑D vector."""
        offset = 0
        for p in self.parameters():
            num = p.numel()
            p.data.copy_(vec[offset:offset+num].view_as(p))
            offset += num

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.net[0].in_features}, hidden_dim={self.net[1].in_features}, output_dim={self.net[-1].out_features}, depth={len(self.net)//2})"
