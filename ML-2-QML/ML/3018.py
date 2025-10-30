"""Hybrid classical classifier/regressor that mirrors the quantum interface.

This module provides a pure‑Python implementation of the same API that
the quantum version exposes.  It is intentionally lightweight so that
experiments can be run on CPU only, while still offering the same
metadata (encoding indices, weight sizes, observables) that the quantum
backend expects.  The class can operate in either classification or
regression mode and can optionally wrap a quantum sub‑module for
hybrid training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List, Union


class HybridQuantumClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        depth: int,
        task: str = "classification",
        use_quantum: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of input features (or qubits for the quantum part).
        depth : int
            Depth of the layered network / ansatz.
        task : {"classification","regression"}
            Which output head to use.
        use_quantum : bool
            If True, the forward method will invoke the quantum sub‑module
            (requires the qml module to be importable).  This keeps the
            classical API identical to the quantum one.
        device : str or torch.device
            Target device.
        """
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.task = task
        self.use_quantum = use_quantum
        self.device = torch.device(device)

        # Build the classical feed‑forward part
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            out_dim = num_features if task == "classification" else 32
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        # Head
        head_out = 2 if task == "classification" else 1
        layers.append(nn.Linear(in_dim, head_out))
        self.classifier = nn.Sequential(*layers)

        # Metadata that mimics the quantum API
        self.encoding_indices: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = [p.numel() for p in self.classifier.parameters()]
        self.observables: List[int] = [0, 1] if task == "classification" else [0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that forwards through the classical network.
        If ``use_quantum`` is true, the call will raise an ImportError
        unless the quantum module is available; the quantum forward
        implementation is provided in the qml module.
        """
        if self.use_quantum:
            raise ImportError("Quantum forward requires the qml module.")
        return self.classifier(x.to(self.device))

    @staticmethod
    def generate_superposition_data(
        num_features: int, samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data that mimics the quantum superposition used in the
        regression seed.  The function is identical to the one in the
        quantum regression example but returns pure NumPy arrays for
        use by the classical model.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a classical feed‑forward network with the same signature
        as the quantum circuit builder.  The return values are:
          - nn.Module: the network
          - encoding indices
          - weight sizes
          - observables
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        net = nn.Sequential(*layers)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in net.parameters()]
        observables = [0, 1]
        return net, encoding, weight_sizes, observables


__all__ = ["HybridQuantumClassifier"]
