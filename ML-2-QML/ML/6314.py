"""
Hybrid fully‑connected layer that can operate in classical or quantum mode.
The classical mode uses a simple nn.Linear with a tanh activation.
The quantum mode evaluates the expectation value of a single‑qubit Ry
parameterized circuit on |0⟩, computed analytically for speed.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


class HybridFCL(nn.Module):
    """
    A hybrid fully‑connected layer that can be configured to use either a
    classical linear mapping or a quantum expectation value.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features for the classical linear layer.
    use_quantum : bool, default=False
        If True, the layer computes the quantum expectation of a Ry(theta)
        circuit on |0⟩ for each theta.  If False, it falls back to a
        classical nn.Linear followed by a tanh activation.
    """

    def __init__(self, n_features: int = 1, use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if not use_quantum:
            self.linear = nn.Linear(n_features, 1)
        else:
            self.linear = None  # placeholder for symmetry

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the layer on a sequence of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter values to evaluate.  For the quantum mode these are
            the angles for Ry gates; for the classical mode they are the
            input scalars fed into the linear layer.

        Returns
        -------
        np.ndarray
            2‑D array of shape (len(thetas), 1) containing the computed
            expectations or activations.
        """
        if self.use_quantum:
            # Quantum expectation of Pauli‑Z after Ry(theta) on |0⟩
            theta = np.array(list(thetas), dtype=np.float32).flatten()
            expectation = np.cos(theta)  # <Z> = cos(theta)
            return expectation.reshape(-1, 1)
        else:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            activation = torch.tanh(self.linear(values)).mean(dim=0)
            return activation.detach().numpy()


__all__ = ["HybridFCL"]
