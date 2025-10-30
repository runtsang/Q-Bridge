import torch
import torch.nn as nn

class QuantumClassifier(nn.Module):
    """
    Classical feed‑forward classifier that mirrors the quantum helper interface.
    Supports optional dropout and batch normalization for richer regularisation.
    """

    def __init__(self, num_features: int, depth: int = 3, dropout: float = 0.1, batch_norm: bool = False):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def weight_sizes(self) -> list[int]:
        return [p.numel() for p in self.parameters()]
