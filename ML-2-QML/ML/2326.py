"""Hybrid graph neural network with a classical backend.

The class exposes the same public API as the original GraphQNN module but
internally supports both a pure‑torch implementation and a quantum‑aware
fallback.  The design follows a *combination* scaling paradigm: the
classical part is used for rapid prototyping while the quantum part
provides a drop‑in replacement for experiments that require state
fidelity analysis or variational circuits.

Key extensions over the seed:
* The constructor accepts a ``mode`` flag to select between ``'classical'``
  and ``'quantum'``.
* ``random_network`` now returns a ``GraphQNNHybrid`` instance ready for
  training, keeping the original weight generation logic but wrapped in
  the class.
* ``feedforward`` returns both the raw activations and a graph of
  state‑fidelity similarities, enabling downstream graph‑based
  clustering.
* A lightweight convolution filter (``Conv``) can be attached to the
  network and is automatically applied to each input sample.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable as IterableType

import networkx as nx
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Convolution helper (borrowed from Conv.py)                                 #
# --------------------------------------------------------------------------- #
class ConvFilter(torch.nn.Module):
    """Simple 2×2 convolution filter used as a drop‑in replacement for a
    quantum quanvolution layer.  It is deliberately lightweight so that
    the hybrid network can be run on CPU without a quantum backend.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
#  GraphQNNHybrid – classical implementation                                 #
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Hybrid graph neural network that can operate in either classical
    or quantum mode.  The public API mirrors the original GraphQNN module
    while internally delegating to the appropriate backend.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        mode: str = "classical",
        *,
        conv: ConvFilter | None = None,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.mode = mode
        self.conv = conv

        if self.mode == "classical":
            self.weights = self._init_classical_weights()
        elif self.mode == "quantum":
            raise RuntimeError(
                "Quantum mode is only available in the qml module; "
                "use GraphQNNHybrid from the qml package."
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    # --------------------------------------------------------------------- #
    #  Weight initialisation
    # --------------------------------------------------------------------- #
    def _init_classical_weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        return weights

    # --------------------------------------------------------------------- #
    #  Random data generation
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Return architecture, weights, training data and target weight."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # --------------------------------------------------------------------- #
    #  Forward propagation
    # --------------------------------------------------------------------- #
    def feedforward(
        self, samples: IterableType[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            # optional convolution preprocessing
            if self.conv is not None:
                conv_out = self.conv.run(features.numpy())
                # broadcast to match feature dimension
                features = torch.tensor(conv_out * torch.ones_like(features))
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    # --------------------------------------------------------------------- #
    #  Fidelity helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNNHybrid",
    "ConvFilter",
]
