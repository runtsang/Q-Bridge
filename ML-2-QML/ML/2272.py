"""Hybrid classical convolution + quantum graph QNN module.

This module defines ConvGraphHybrid that combines a 2‑D convolution
filter with a quantum graph‑based neural network.  The classical
filter produces a single activation per patch; the quantum part
encodes the patch, propagates it through a random unitary network,
and builds a fidelity‑based graph.  The output is a tuple
(classical_activation, graph).

The design draws from Conv.py (classical filter) and
GraphQNN.py (state propagation & fidelity graph).  The quantum
computation is delegated to the quantum module defined in
`quantum_graph_qnn.py`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical convolution filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolution that mimics the quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return a single activation per patch."""
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# --------------------------------------------------------------------------- #
#  Quantum graph QNN utilities
# --------------------------------------------------------------------------- #
# The heavy quantum code lives in the separate module `quantum_graph_qnn`.
# We import the public function that returns the fidelity graph.
from quantum_graph_qnn import run_quantum_graph

# --------------------------------------------------------------------------- #
#  Hybrid forward pass
# --------------------------------------------------------------------------- #
class ConvGraphHybrid(nn.Module):
    """
    Hybrid module that runs a classical convolution followed by a
    quantum graph‑based QNN.  The forward method accepts a 2‑D image
    patch and returns a tuple ``(classical_activation, graph)``.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        qnn_arch: Sequence[int] = (2, 2, 2),
        graph_threshold: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__()
        self.classical = ConvFilter(kernel_size=kernel_size, threshold=0.0)
        self.qnn_arch = list(qnn_arch)
        self.graph_threshold = graph_threshold
        # Pass-through arguments for the quantum routine
        self.qnn_kwargs = kwargs

    def forward(self, x: Tensor) -> Tuple[Tensor, nx.Graph]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, 1, H, W) where H and W equal
            ``kernel_size``.  The module processes each batch element
            independently.

        Returns
        -------
        Tuple[Tensor, nx.Graph]
            * Classical activation of shape (B, 1, 1, 1).
            * Fidelity‑based adjacency graph constructed from the
              quantum states produced by the QNN.
        """
        # Classical convolution
        classical_out = self.classical(x)

        # Flatten each patch to a 1‑D array for the quantum routine.
        # We take the mean over the batch dimension to keep the
        # implementation simple; for a full batch version one could
        # loop over batch elements.
        batch_mean = classical_out.mean(dim=(0, 1, 2)).detach().cpu().numpy()

        # The quantum routine expects a 1‑D array of length 2**n.
        # Pad or truncate to the nearest power of two.
        target_len = 2 ** (len(self.qnn_arch) - 1)
        if batch_mean.size < target_len:
            padded = np.pad(batch_mean, (0, target_len - batch_mean.size), "constant")
        else:
            padded = batch_mean[:target_len]

        # Run the quantum graph QNN and obtain the adjacency graph.
        graph = run_quantum_graph(
            data=padded,
            qnn_arch=self.qnn_arch,
            threshold=self.graph_threshold,
            **self.qnn_kwargs,
        )

        return classical_out, graph

__all__ = ["ConvGraphHybrid"]
