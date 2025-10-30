import torch
from torch import nn

class FraudDetectionHybrid(nn.Module):
    """
    Classical head that consumes quantum‑derived features.
    Architecture mirrors the EstimatorQNN regressor but accepts
    a 3‑dimensional input: two photonic quadrature expectations
    and one Qiskit Y‑observable expectation.
    """
    def __init__(self, input_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
