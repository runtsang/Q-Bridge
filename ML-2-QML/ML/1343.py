"""Enhanced feed‑forward regressor with residual connections and dropout for regression tasks."""

import torch
from torch import nn

class EstimatorNN(nn.Module):
    """A deep residual network for regression with dropout and batch normalisation."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.input_layer(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = out + residual  # residual connection
        out = self.output_layer(out)
        return out

def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the enhanced regression network."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
