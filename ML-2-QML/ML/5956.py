import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """
    An expressive classical sampler network.
    Extends the original 2‑layer architecture with optional
    batch‑normalization and dropout for improved regularisation.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 2,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities with softmax.
        """
        return F.softmax(self.net(x), dim=-1)

def SamplerQNN() -> SamplerModule:
    """
    Factory returning a ready‑to‑use SamplerModule instance.
    """
    return SamplerModule()

__all__ = ["SamplerQNN"]
