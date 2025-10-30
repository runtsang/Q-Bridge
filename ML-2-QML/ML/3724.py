import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class UnifiedQNN(nn.Module):
    """
    Hybrid classical network that mirrors the SamplerQNN and EstimatorQNN seeds.
    It contains:
        • A lightweight “quantum” feature extractor (here a single linear layer).
        • A classical post‑processing head that can be a soft‑max classifier (sampler)
          or a regression head (estimator).
    The design keeps the torch API unchanged while allowing a quantum backend to
    plug in later.
    """

    def __init__(self,
                 mode: str = "sampler",
                 input_dim: int = 2,
                 quantum_dim: int = 2) -> None:
        """
        Parameters
        ----------
        mode : {"sampler", "estimator"}
            Determines whether the model behaves like a SamplerQNN
            (soft‑max over two classes) or an EstimatorQNN (regression).
        input_dim : int
            Dimensionality of the classical input vector.
        quantum_dim : int
            Dimensionality of the quantum feature space (default 2).
        """
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim

        # Classical “quantum” extractor: a simple linear map
        self.quantum_layer = nn.Linear(input_dim, quantum_dim, bias=False)

        # Classical post‑processing head
        if mode == "sampler":
            self.head = nn.Sequential(
                nn.Linear(quantum_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 2)
            )
        elif mode == "estimator":
            self.head = nn.Sequential(
                nn.Linear(quantum_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1)
            )
        else:
            raise ValueError("mode must be'sampler' or 'estimator'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (..., input_dim)

        Returns:
            Tensor of shape (..., 2) for sampler mode (soft‑max probabilities)
            or (..., 1) for estimator mode (regression output).
        """
        # Simple classical surrogate for quantum feature extraction
        quantum_features = self.quantum_layer(x)

        # Classical post‑processing
        out = self.head(quantum_features)

        if self.mode == "sampler":
            return F.softmax(out, dim=-1)
        return out

__all__ = ["UnifiedQNN"]
