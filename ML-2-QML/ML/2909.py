"""Hybrid fully‑connected layer with a classical surrogate for the quantum part."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

class QuantumLayerSim(nn.Module):
    """
    Classical surrogate for a quantum layer.
    Computes a differentiable expectation value as a simple function
    of the variational parameters.
    """
    def __init__(self, num_params: int):
        super().__init__()
        # Treat each parameter as a learnable weight; initialise to zero.
        self.weights = nn.Parameter(torch.zeros(num_params))

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation value as weighted sum of sin(thetas).
        Parameters
        ----------
        thetas : torch.Tensor
            Tensor of shape (num_params,) containing variational parameters.
        Returns
        -------
        torch.Tensor
            Scalar expectation value in range [-1, 1].
        """
        # Ensure thetas are on the same device as weights
        thetas = thetas.to(self.weights.device)
        expectation = torch.sum(self.weights * torch.sin(thetas))
        return expectation

class HybridFCLClassifier(nn.Module):
    """
    Hybrid classical‑quantum classifier.
    Combines a linear feature extractor, a quantum surrogate layer,
    and a final classification head.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_qubits: int, depth: int, num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden_dim)
        self.quantum = QuantumLayerSim(num_qubits * depth)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, num_features).
        thetas : torch.Tensor
            Variational parameters of shape (num_qubits * depth,).
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        h = torch.relu(self.encoder(x))
        q_out = self.quantum(thetas)
        # Broadcast quantum output to batch dimension
        q_out = q_out.expand_as(h)
        combined = h + q_out
        logits = self.classifier(combined)
        return logits

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Convenience method mimicking the original FCL interface.
        Returns the expectation value of the quantum surrogate.
        Parameters
        ----------
        thetas : np.ndarray
            Array of variational parameters.
        Returns
        -------
        np.ndarray
            Array containing a single expectation value.
        """
        with torch.no_grad():
            theta_tensor = torch.tensor(thetas, dtype=torch.float32)
            expectation = self.quantum(theta_tensor).cpu().numpy()
        return np.array([expectation])

__all__ = ["HybridFCLClassifier"]
