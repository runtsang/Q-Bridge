import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """Classical fully connected layer with optional dropout and gating.

    The layer accepts a variable number of parameters ``thetas``,
    feeds them through a linear transformation followed by a tanh
    activation, and returns the mean activation as a NumPy array.
    Dropout and gating can be toggled to explore overfitting
    and expressivity trade‑offs.
    """

    def __init__(self, n_features: int = 1, dropout: float = 0.0, use_gate: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(n_features, 1, bias=False)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        x = self.linear(theta_tensor)
        x = self.dropout(x)
        out = torch.tanh(x)
        if self.use_gate:
            gate = torch.sigmoid(self.gate(theta_tensor))
            out = out * gate
        return out.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the layer and return a NumPy array."""
        with torch.no_grad():
            result = self.forward(thetas)
        return result.detach().cpu().numpy()
