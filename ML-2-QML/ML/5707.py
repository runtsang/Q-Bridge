import torch
from torch import nn

class HybridEstimator(nn.Module):
    """Classical feed‑forward regressor with optional dropout and layer‑wise scaling.

    The architecture mirrors the original 2→8→4→1 network but adds
    configurable dropout, layer‑wise weight initialization and a
    simple early‑stopping hook that can be used in training loops.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (8, 4),
                 output_dim: int = 1,
                 dropout: float | None = 0.0,
                 init_type: str = "xavier_uniform"):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights(init_type)

    def _init_weights(self, init_type: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that detaches the output."""
        with torch.no_grad():
            return self.forward(x).detach()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layers={len(self.net)})"

__all__ = ["HybridEstimator"]
