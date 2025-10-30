import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """
    A depth‑adaptive, dropout‑regularised convolutional neural network that
    mirrors the structure of a quantum convolutional network.  The network
    can be configured with arbitrary width, depth, and dropout probability,
    and supports batch‑normalisation between layers for stable training.
    """

    def __init__(
        self,
        input_dim: int = 8,
        widths: list[int] | None = None,
        depth: int = 3,
        dropout: float = 0.2,
        device: str | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.dropout = dropout
        # Default widths follow the original seed but are truncatable by depth
        widths = widths or [16, 16, 8, 4, 4]
        self.layers = nn.ModuleList()
        prev = input_dim
        for w in widths[:depth]:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev, w),
                    nn.BatchNorm1d(w),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )
            prev = w
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions in inference mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(logits, targets)

    def summary(self) -> None:
        """Print a concise summary of the architecture."""
        print(f"QCNNHybrid depth={self.depth} widths={len(self.layers)} dropout={self.dropout}")
        for i, layer in enumerate(self.layers):
            print(f"  layer {i}: {layer}")
