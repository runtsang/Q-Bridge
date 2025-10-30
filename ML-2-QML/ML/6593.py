import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A lightweight residual block that keeps the input dimension unchanged."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QCNNModel(nn.Module):
    """
    A deeper, more expressive QCNN-inspired network.

    The architecture follows the original 8‑bit feature‑map size but
    expands to multiple stages, using *skip‑connections* and
    **in‑place** post‑processing for the final sigmoid.
    """
    def __init__(self, depth: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.Tanh()
        )
        self.stages = nn.ModuleList()
        for _ in range(depth):
            self.stages.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
                )
            )
        self.residual = ResidualBlock(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        for stage in self.stages:
            x = stage(x)
        x = self.residual(x)
        x = self.pool(x.unsqueeze(1)).squeeze(-1)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning a fully‑configured, deep QCNN‑like model."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
