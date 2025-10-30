from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """Simple residual block with linear layer, batch norm, ReLU, and dropout."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + residual


class QuantumClassifierModel:
    """Classical feed‑forward classifier that mirrors the quantum interface."""
    def __init__(self, num_features: int, depth: int, dropout: float = 0.1):
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.network, self.weight_sizes = self._build_network()
        self.encoding = list(range(num_features))
        self.observables = [0, 1]  # class indices for compatibility

    def _build_network(self) -> Tuple[nn.Module, List[int]]:
        layers = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(ResidualBlock(in_dim, self.dropout))
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        network = nn.Sequential(*layers)
        weight_sizes = [
            m.weight.numel() + m.bias.numel()
            for m in network.modules()
            if isinstance(m, nn.Linear)
        ]
        return network, weight_sizes

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        verbose: bool = False,
    ) -> None:
        """Standard training loop with Adam optimizer."""
        self.network.to(device)
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.network.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss/len(dataset):.4f}"
                )

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return class predictions."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(device))
            return torch.argmax(logits, dim=1)

    def get_encoding(self) -> List[int]:
        """Return the feature indices used by the model."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per linear layer."""
        return self.weight_sizes

    def get_observables(self):
        """Return a placeholder list of observables for API compatibility."""
        return self.observables


__all__ = ["QuantumClassifierModel"]
