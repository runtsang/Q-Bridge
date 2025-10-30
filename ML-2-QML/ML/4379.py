import torch
from torch import nn
import numpy as np

class QuantumInspiredLayer(nn.Module):
    """Classical layer mimicking a quantum expectation value."""
    def __init__(self, n_params: int, n_outputs: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_params, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input to parameters
        params = self.linear(x)
        # Simulate expectation: tanh of params, mean over batch
        expectation = torch.tanh(params).mean(dim=0, keepdim=True)
        return expectation

class HybridEstimatorQNN(nn.Module):
    """Hybrid classical model combining linear feature extraction,
    a quantum‑inspired layer, and a final regression head."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, n_q_params: int = 4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.quantum_layer = QuantumInspiredLayer(n_params=n_q_params, n_outputs=1)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        q_out = self.quantum_layer(feats)
        out = self.head(q_out)
        return out

__all__ = ["HybridEstimatorQNN"]
