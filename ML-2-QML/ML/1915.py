import torch
from torch import nn

class EstimatorQNNGen(nn.Module):
    """
    Enhanced fully‑connected regressor with configurable depth,
    dropout, and batch‑normalisation.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (8, 4),
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        activation: nn.Module = nn.Tanh(),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

def EstimatorQNN() -> EstimatorQNNGen:
    """
    Factory returning a default‑configured instance.
    """
    return EstimatorQNNGen(
        hidden_dims=(12, 8, 4),
        dropout_rate=0.15,
        seed=42,
    )

__all__ = ["EstimatorQNNGen", "EstimatorQNN"]
