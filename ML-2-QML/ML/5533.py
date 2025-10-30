import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class HybridFCL(nn.Module):
    """
    Classical approximation of a hybrid fully connected layer.
    It emulates the behaviour of a quantum parameterised circuit
    by mapping the input through a linear encoder, a small neural
    network that acts as a quantum layer and a final linear head.
    """

    def __init__(self, n_features: int = 1, n_qubits: int = 1, hidden_units: int = 8) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits

        # Linear encoder that projects the raw input into a 1‑D space
        self.encoder = nn.Linear(n_features, 1) if n_features > 1 else nn.Identity()

        # Classical “quantum” layer: a small feed‑forward network
        self.quantum_approx = nn.Sequential(
            nn.Linear(1, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )

        # Classical head that produces the final regression output
        self.head = nn.Linear(1, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that accepts an iterable of theta values
        (one per sample).  The values are first encoded, then
        processed by the quantum‑approximation network and finally
        mapped to a scalar output.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        encoded = self.encoder(theta_tensor)
        q_expect = self.quantum_approx(encoded)
        output = self.head(q_expect)
        return output.squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that runs the module in evaluation
        mode and returns a NumPy array of predictions.
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(thetas).cpu().numpy()
        return preds

__all__ = ["HybridFCL"]
