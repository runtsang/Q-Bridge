"""Hybrid classical encoder combining a fully connected layer and a sampler-like network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCLSampler(nn.Module):
    """
    Classical module that prepares parameters for a quantum sampler.
    Combines a fully‑connected layer (from the FCL seed) and a two‑layer
    feed‑forward network (from the SamplerQNN seed). The output tensor
    contains parameters to bind to the quantum circuit.
    """
    def __init__(self, n_features: int = 1, n_qubits: int = 2, n_weights: int = 4) -> None:
        super().__init__()
        # Fully‑connected layer producing a single output per feature
        self.fc = nn.Linear(n_features, 1)
        # Sampler‑style network producing a 2‑dimensional probability vector
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.n_qubits = n_qubits
        self.n_weights = n_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Produces a parameter vector of shape (n_qubits + n_weights,) that
        can be bound to the quantum circuit. The vector is constructed by
        concatenating the sampler output with a placeholder for the remaining
        weights (which will be set by the quantum backend).
        """
        # First linear transform
        fc_out = torch.tanh(self.fc(x)).view(-1)
        # Dummy two‑dim input for the sampler network
        dummy = torch.stack([fc_out, fc_out], dim=-1)
        probs = F.softmax(self.sampler_net(dummy), dim=-1)
        # Concatenate to form the full parameter list
        params = torch.cat([probs.squeeze(), torch.zeros(self.n_qubits + self.n_weights - probs.size(0))])
        return params

__all__ = ["HybridFCLSampler"]
