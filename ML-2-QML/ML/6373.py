import torch
from torch import nn
import numpy as np

class HybridFCL(nn.Module):
    """
    Classical hybrid model combining a feed‑forward regressor (EstimatorQNN style)
    with a quantum‑inspired rotation layer (FCL style). The forward pass first
    maps 2‑D inputs through a small neural net, then interprets the scalar
    output as a rotation angle θ. The model returns the expectation value
    of σ_y on a single qubit prepared in |+⟩ and rotated by Ry(θ).
    This analytic expectation is sin(θ), which matches the quantum
    implementation in the QML counterpart.
    """
    def __init__(self) -> None:
        super().__init__()
        # EstimatorQNN architecture
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Linear layer to map network output to rotation angle range
        self.theta_mapper = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): shape (..., 2) – two‑dimensional input features.

        Returns:
            torch.Tensor: shape (..., 1) – expectation value of σ_y.
        """
        raw = self.net(inputs)          # (..., 1)
        theta = self.theta_mapper(raw)  # (..., 1)
        # Quantum‑inspired expectation: sin(θ)
        expectation = torch.sin(theta)
        return expectation

# Expose for import
__all__ = ["HybridFCL"]
