import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier that mimics the interface of the quantum
    counterpart.  The network is fully differentiable in PyTorch and can be
    trained with any optimiser.

    Parameters
    ----------
    num_features : int
        Number of input features (also the size of the hidden layers).
    depth : int
        Number of hidden layers.
    """

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

        # Metadata that mirrors the quantum implementation
        self.encoding = list(range(num_features))
        self.weight_sizes = [
            w.numel() + b.numel()
            for w, b in zip(self.net.children()[: depth * 2: 2], self.net.children()[1: depth * 2: 2])
        ]
        self.weight_sizes.append(self.net[-1].weight.numel() + self.net[-1].bias.numel())
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_parameter_vector(self) -> torch.Tensor:
        """Return a flat vector of all learnable parameters."""
        return torch.cat([p.view(-1) for p in self.parameters()])

    def set_parameter_vector(self, vec: torch.Tensor) -> None:
        """Set the model parameters from a flat vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(vec[offset : offset + numel].view_as(p))
            offset += numel

    def train_step(self, x: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer) -> float:
        """Single optimisation step."""
        opt.zero_grad()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        return loss.item()

__all__ = ["QuantumClassifierModel"]
