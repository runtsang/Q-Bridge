"""Enhanced feed‑forward regressor with regularization and flexible architecture."""
import torch
from torch import nn

class EstimatorNN(nn.Module):
    """
    A small MLP that adds batch‑normalization, dropout, and a configurable hidden
    layer stack.  Designed for regression tasks with arbitrary input dimensionality.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] | list[int] = (8, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference in evaluation mode with no gradients.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def EstimatorQNN() -> EstimatorNN:
    """
    Return an instance of the extended EstimatorNN.
    """
    return EstimatorNN()
