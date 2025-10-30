import torch
from torch import nn
import torch.nn.functional as F


class EstimatorQNNGen208(nn.Module):
    """
    A flexible fully‑connected regression network that generalises the original
    EstimatorQNN.  The architecture is driven by a *config* dictionary that can
    specify the number of hidden layers, their dimensionality, whether to
    interleave dropout and batch‑norm, and the weight‑initialisation scheme.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        in_features = cfg.get("in_features", 2)
        hidden_layers = cfg.get("hidden_layers", [8, 4])
        out_features = cfg.get("out_features", 1)
        dropout = cfg.get("dropout", 0.0)
        batch_norm = cfg.get("batch_norm", False)
        init = cfg.get("init", "xavier_uniform")

        layers = []
        prev = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.model = nn.Sequential(*layers)

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that disables gradients."""
        with torch.no_grad():
            return self.forward(x)


__all__ = ["EstimatorQNNGen208"]
