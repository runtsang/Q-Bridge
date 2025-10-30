import torch
import torch.nn as nn
import numpy as np

class HybridFCL(nn.Module):
    """
    Hybrid fully connected layer that combines a classical linear transformation
    with a quantum feature map. The linear layer produces parameters for the
    quantum circuit, which returns an expectation value used as the layer output.
    """
    def __init__(self, n_features: int, quantum_layer) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.quantum_layer = quantum_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: linear -> quantum feature map.

        Args:
            x: Tensor of shape (batch, n_features)

        Returns:
            Tensor of shape (batch, 1) containing the quantum expectation values.
        """
        # Linear output
        lin_out = self.linear(x)  # shape (batch, 1)
        # Convert to numpy for quantum layer
        thetas = lin_out.detach().cpu().numpy().squeeze()
        # Quantum expectation
        q_out = self.quantum_layer.run(thetas)  # shape (batch, 1)
        return torch.from_numpy(q_out).float()

__all__ = ["HybridFCL"]
