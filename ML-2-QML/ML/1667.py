import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple, List

class QuantumClassifierModel(nn.Module):
    """Classical feed‑forward classifier mirroring the quantum helper interface.

    The class extends the seed by exposing a transparent training API,
    weight‑size introspection, and a convenience method that returns the
    same metadata (encoding, weight_sizes, observables) the quantum
    counterpart expects.
    """
    def __init__(self, num_features: int, depth: int, device: str = "cpu"):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.device = torch.device(device)
        self.net, self.encoding, self.weight_sizes, self.observables = self._build()
        self.to(self.device)

    def _build(self) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        layers = []
        in_dim = self.num_features
        encoding = list(range(self.num_features))
        weight_sizes = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        net = nn.Sequential(*layers)
        observables = list(range(2))
        return net, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: optim.Optimizer,
                   criterion: nn.Module) -> float:
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y).float().mean().item()
            loss = nn.functional.cross_entropy(logits, y).item()
        return loss, accuracy

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding, weight_sizes and observables so that the quantum
        interface can consume the same structure."""
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
