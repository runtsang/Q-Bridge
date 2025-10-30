"""GraphQNNGen397: hybrid classical‑quantum graph neural network.

This module implements a lightweight hybrid interface that mirrors the
original GraphQNN utilities while delegating to quantum back‑ends
when available.  It exposes wrappers for a quantum EstimatorQNN,
a quantum LSTM, and a quantum convolutional filter, enabling
researchers to experiment with classical and quantum components
in a single pipeline.
"""

from __future__ import annotations

import itertools
import networkx as nx
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# Optional quantum helpers
try:
    from.EstimatorQNN import EstimatorQNN
except Exception:
    EstimatorQNN = None

try:
    from.QLSTM import QLSTM
except Exception:
    QLSTM = None

try:
    from.Conv import Conv
except Exception:
    Conv = None

Tensor = torch.Tensor


class GraphQNNGen397:
    """Hybrid graph neural network that can run in classical or quantum mode."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights: List[Tensor] = [
            self._rand_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])
        ]
        self._estimator = EstimatorQNN() if EstimatorQNN else None
        self._lstm_cls = QLSTM if QLSTM else nn.LSTM
        self._conv_cls = Conv if Conv else None

    @staticmethod
    def _rand_linear(in_f: int, out_f: int) -> Tensor:
        """Generate a random weight matrix."""
        return torch.randn(out_f, in_f, dtype=torch.float32)

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int) -> "GraphQNNGen397":
        """Instantiate a network with random weights and a simple training set."""
        instance = cls(arch)
        target_weight = instance.weights[-1]
        data = []
        for _ in range(samples):
            xs = torch.randn(target_weight.size(1))
            ys = target_weight @ xs
            data.append((xs, ys))
        return instance, data

    def feedforward(self, inputs: Tensor) -> List[List[Tensor]]:
        """Compute activations layer‑wise for each input sample."""
        activations: List[List[Tensor]] = []
        for sample in inputs:
            layer_vals: List[Tensor] = [sample]
            current = sample
            for w in self.weights:
                current = torch.tanh(w @ current)
                layer_vals.append(current)
            activations.append(layer_vals)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Normalized squared inner product."""
        an = a / (torch.norm(a) + 1e-12)
        bn = b / (torch.norm(b) + 1e-12)
        return float((an @ bn).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g

    # Quantum‑aware wrappers ------------------------------------------------

    def run_estimator(self, data: Tensor) -> Tensor:
        """Return the quantum EstimatorQNN object if available."""
        if self._estimator is None:
            raise RuntimeError("EstimatorQNN module not available")
        return self._estimator

    def run_lstm(self, inputs: Tensor, n_qubits: int = 0) -> Tensor:
        """Run a quantum or classical LSTM on a sequence."""
        lstm = self._lstm_cls(self.arch[0], self.arch[-1], n_qubits=n_qubits)
        output, _ = lstm(inputs)
        return output

    def run_conv(self, data: np.ndarray) -> float:
        """Apply a quantum convolution filter to a 2‑D array."""
        if self._conv_cls is None:
            raise RuntimeError("Conv module not available")
        filt = self._conv_cls()
        return filt.run(data)


__all__ = ["GraphQNNGen397"]
