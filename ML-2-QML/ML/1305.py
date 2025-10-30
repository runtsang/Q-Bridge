"""QCNN: Residual, batch‑normed, and dropout‑regularised classical convolutional network.

This class extends the original seed by adding residual connections, batch‑normalisation and dropout, providing a richer feature extractor that is more robust to over‑fitting.  The architecture is intentionally designed to mirror the depth and dimensionality of the quantum version, facilitating fair comparisons in hybrid experiments.
"""

import torch
from torch import nn

class QCNN(nn.Module):
    """Residual QCNN with batch‑norm and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        # Convolutional layers with residual shortcuts
        self.conv_layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            # Shortcut from the initial feature map to each conv layer
            self.residuals.append(nn.Linear(hidden_dims[0], hidden_dims[i]))
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x0 = self.feature_map(x)
        out = x0
        for conv, res in zip(self.conv_layers, self.residuals):
            out = conv(out) + res(x0)
        return torch.sigmoid(self.head(out))

def QCNN() -> QCNN:
    """Factory returning a ready‑to‑train QCNN instance."""
    return QCNN()

__all__ = ["QCNN", "QCNN"]
