"""Combined classical graph neural network and variational classifier.

This module merges the graph‑based feed‑forward utilities from the original
GraphQNN seed with the shallow neural‑network classifier from
QuantumClassifierModel.  All public methods expose a common
interface so that the same experiment can be run on a classical
backend or on a quantum simulator.

Core features
* Randomly generated weight matrices and training data.
* Feed‑forward propagation with tanh activations.
* Fidelity‑based graph construction for state similarity.
* A convenience factory for a shallow two‑class classifier.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


class GraphQNNClassifier:
    """Dual‑mode Graph‑QNN classifier.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the graph‑based neural network.
    depth : int
        Depth of the shallow classifier head.
    num_qubits : int
        Number of qubits / input features for the quantum side.
    """

    def __init__(self, qnn_arch: Sequence[int], depth: int, num_qubits: int):
        self.qnn_arch = list(qnn_arch)
        self.depth = depth
        self.num_qubits = num_qubits

    # --------------------------------------------------------------------- #
    #  Classical utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random weight chain and a training set for the last layer."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            weights.append(self._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return self.qnn_arch, weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Produce (x, Wx) pairs for a given linear map."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=torch.float32)
            y = weight @ x
            dataset.append((x, y))
        return dataset

    def feedforward(self, weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass through the weight chain."""
        outputs: List[List[Tensor]] = []
        for x, _ in samples:
            activations: List[Tensor] = [x]
            h = x
            for w in weights:
                h = torch.tanh(w @ h)
                activations.append(h)
            outputs.append(activations)
        return outputs

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two classical vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph where edges encode state fidelity."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # --------------------------------------------------------------------- #
    #  Classifier factory
    # --------------------------------------------------------------------- #

    def build_classifier(self) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """Return a shallow feed‑forward classifier and metadata."""
        layers: List[nn.Module] = []
        in_dim = self.num_qubits
        encoding: List[int] = list(range(self.num_qubits))
        weight_sizes: List[int] = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_qubits)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_qubits

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        net = nn.Sequential(*layers)
        observables: List[int] = list(range(2))
        return net, encoding, weight_sizes, observables
