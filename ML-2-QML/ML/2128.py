import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, List, Tuple

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward network that mirrors the API of the quantum helper.
    Supports training via Adam, prediction, and metadata extraction.
    """
    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        layers = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = hidden_dim

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables = list(range(2))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            lr: float = 1e-3, epochs: int = 50,
            device: str | torch.device = "cpu") -> None:
        """
        Train the network using a simple Adam optimiser.
        """
        self.to(device)
        X, y = X.to(device), y.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        X = X.to(device)
        with torch.no_grad():
            logits = self.forward(X)
        return logits.argmax(dim=-1)

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Return encoding, weight sizes, and observables to match the quantum interface.
        """
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
