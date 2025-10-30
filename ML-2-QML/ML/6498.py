import torch
from torch import nn
import numpy as np
from typing import Iterable

class HybridEstimatorQNN(nn.Module):
    """
    Combines a quantum‑inspired feature extractor with a classical
    regression head.  The feature extractor emulates a 1‑qubit circuit
    with input and weight parameters, computing the analytical
    expectation of the Pauli‑Y observable.  This allows end‑to‑end
    differentiability without a quantum backend.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        # Parameters that would correspond to quantum angles
        self.theta = nn.Parameter(torch.randn(input_dim))
        self.phi = nn.Parameter(torch.randn(hidden_dim))
        # Classical regression network
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def _quantum_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical expectation value of Y for a 1‑qubit
        circuit with input rotation Ry(θ) and trainable rotation Rx(ϕ).
        For a single qubit the expectation is 2 * sin(θ) * cos(ϕ).
        """
        theta = torch.sum(x * self.theta, dim=-1)  # shape (batch, hidden_dim)
        exp_val = 2 * torch.sin(theta) * torch.cos(self.phi)  # broadcast phi
        return exp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map each input sample to a vector of quantum
        expectation values (length = hidden_dim) and feed it through
        the classical regression head.
        """
        batch = x.shape[0]
        # Expand input to match hidden_dim
        x_expanded = x.unsqueeze(1).repeat(1, self.phi.shape[0], 1)
        q_features = self._quantum_expectation(x_expanded)
        return self.regressor(q_features)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the `run` method of a quantum circuit.  Accepts a list of
        parameters and returns the single expectation value as a NumPy
        array.  Useful for unit‑testing the quantum‑inspired layer.
        """
        with torch.no_grad():
            theta_tensor = torch.tensor(thetas, dtype=torch.float32)
            exp_val = self._quantum_expectation(theta_tensor)
            return exp_val.detach().numpy()

__all__ = ["HybridEstimatorQNN"]
