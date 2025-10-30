"""Hybrid sampler‑classifier (classical side).

This module implements the classical component of the hybrid architecture.
It mirrors the SamplerQNN network, extends it with an additional hidden layer,
and replaces the quantum expectation with a linear surrogate (FCL).
A ``mode`` flag allows switching to a quantum backend if desired.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridSamplerClassifier(nn.Module):
    """
    Classical feed‑forward sampler and classifier with a quantum‑style
    expectation surrogate.

    Parameters
    ----------
    n_features : int, default=2
        Dimensionality of the input features.
    hidden_dim : int, default=4
        Width of the intermediate feature extractor.
    n_qubits : int, default=2
        Number of qubits that the quantum layer would act on.
    depth : int, default=1
        Depth of the variational circuit (ignored in classical mode).
    mode : str, default='classical'
        'classical' uses a linear surrogate; 'quantum' expects an external
        quantum backend to evaluate the circuit.
    """

    def __init__(
        self,
        n_features: int = 2,
        hidden_dim: int = 4,
        n_qubits: int = 2,
        depth: int = 1,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.mode = mode

        # Classical encoder (extends SamplerQNN with an extra hidden layer)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Linear surrogate of the quantum expectation (FCL)
        self.expectation_layer = nn.Linear(n_qubits, 1)

        # Classification head
        self.classifier = nn.Linear(1, 2)

        # Variational weights (one per qubit per depth layer)
        self.weight_params = nn.Parameter(
            torch.randn(n_qubits * depth, requires_grad=True)
        )

    def run_quantum_expectation(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation value for a given encoding.

        In classical mode this uses the linear surrogate.
        In quantum mode the call is a no‑op and should be replaced by an
        external circuit evaluation.
        """
        if self.mode == "classical":
            expectation = torch.tanh(self.expectation_layer(encoding))
            return expectation.mean(dim=0, keepdim=True)
        else:
            raise RuntimeError(
                "Quantum mode requires external circuit evaluation."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        encoded = self.encoder(x)  # (batch, hidden_dim)
        # Map encoded features to a vector of size n_qubits
        encoding = encoded[:, : self.expectation_layer.in_features]
        expectation = self.run_quantum_expectation(encoding)
        logits = self.classifier(expectation)
        return logits

    def set_mode(self, mode: str) -> None:
        """Switch between 'classical' and 'quantum' evaluation."""
        if mode not in ("classical", "quantum"):
            raise ValueError("mode must be 'classical' or 'quantum'")
        self.mode = mode


def SamplerQNN() -> nn.Module:
    """Compatibility wrapper that returns the hybrid model."""
    return HybridSamplerClassifier()


__all__ = ["HybridSamplerClassifier", "SamplerQNN"]
