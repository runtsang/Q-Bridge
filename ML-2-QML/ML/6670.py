import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

class QuantumClassifierModel:
    """
    Classical feed‑forward classifier that mirrors the quantum helper interface.
    The network is composed of `depth` blocks of Linear → ReLU layers with optional
    dropout.  The class returns the same metadata (encoding indices, weight sizes,
    and observables) as the quantum variant so that experiments can seamlessly
    swap implementations.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        device: str | None = None
    ):
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.device = torch.device(device or "cpu")

        # Build the network
        self.network = self._build_network().to(self.device)

        # Metadata
        self.encoding = list(range(num_features))
        self.weight_sizes = self._compute_weight_sizes()
        self.observables = [0, 1]  # placeholder to match quantum interface

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self) -> List[int]:
        sizes = []
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module
    ) -> float:
        self.network.train()
        logits = self.forward(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
