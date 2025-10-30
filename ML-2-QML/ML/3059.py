"""Hybrid estimator that couples a classical linear head to a quantum layer.

The module is intentionally free of quantum imports; the quantum
evaluation is supplied by a callable that implements a `run` method.
This design keeps the ML side purely classical while still
exposing the full expressive power of the quantum sub‑network.

Author: gpt-oss-20b
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable

class HybridEstimator(nn.Module):
    """
    A lightweight neural network that stacks:
        1. A linear feature extractor.
        2. A quantum fully‑connected layer (provided as a callable).
        3. A final linear output head.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    quantum_layer : Callable[[np.ndarray], np.ndarray]
        Quantum sub‑network that accepts a 2‑D numpy array of shape
        (batch_size, n_params) and returns a 1‑D array of expectation
        values, one per batch element.
    """
    def __init__(self, n_features: int, quantum_layer: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(n_features, 1)
        self.quantum_layer = quantum_layer
        self.output_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        # Classical feature extraction
        params = self.feature_extractor(x)  # shape: (batch, 1)
        # Detach and move to CPU for quantum evaluation
        params_np = params.detach().cpu().numpy()
        # Run quantum layer
        q_out_np = self.quantum_layer(params_np)  # shape: (batch,)
        q_out = torch.from_numpy(q_out_np).float().to(x.device).unsqueeze(-1)
        # Classical output head
        return self.output_head(q_out)

__all__ = ["HybridEstimator"]
