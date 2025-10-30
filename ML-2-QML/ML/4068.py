"""
Hybrid convolution module – classical implementation.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Classical convolution filter with optional quantum augmentation
# --------------------------------------------------------------------------- #
class HybridConvFilter(nn.Module):
    """
    Classical convolution filter that can be paired with a quantum sub‑module.
    The filter consists of a single learnable 2‑D kernel followed by a
    sigmoid activation.  The output can be enriched by the scalar result
    of a quantum circuit via `set_quantum_filter`.
    """

    def __init__(self, kernel_size: int = 3, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.q_output: nn.Module | None = None

    def set_quantum_filter(self, qfilter: nn.Module) -> None:
        """Attach a quantum filter that provides a scalar output."""
        self.q_output = qfilter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        cls = self.conv(x)
        cls_act = torch.sigmoid(cls - self.conv_threshold)
        cls_mean = cls_act.mean(dim=[2, 3], keepdim=True)

        # Quantum path – if attached
        if self.q_output is None:
            return cls_mean

        # Convert the batch to numpy for the quantum routine
        q_val = self.q_output.run(x.squeeze(0).cpu().numpy())
        q_tensor = torch.tensor(q_val, dtype=cls.dtype, device=cls.device).view(1, 1, 1, 1)
        return cls_mean + q_tensor

    def fidelity_graph(
        self,
        samples: list[torch.Tensor],
        threshold: float = 0.8,
    ) -> nx.Graph:
        """
        Build a graph whose nodes are the quantum outputs of the filter
        applied to each sample.  Edges are added when the absolute
        difference between two outputs exceeds `threshold`.  This simple
        proxy for state fidelity can guide adaptive threshold tuning.
        """
        outputs = [self.q_output.run(s.squeeze(0).cpu().numpy()) for s in samples]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(outputs)))
        for (i, a), (j, b) in itertools.combinations(enumerate(outputs), 2):
            fid = np.abs(a - b)  # scalar fidelity proxy
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
        return graph


# --------------------------------------------------------------------------- #
#  Graph‑based utilities – adapted from the original GraphQNN seed
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: list[int], samples: int):
    """
    Generate a random sequence of weight matrices for a simple feed‑forward
    network.  The last layer is treated as the “target” and is used to
    produce training data via a linear mapping.
    """
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = [
        (torch.randn(target_weight.size(1)), target_weight @ torch.randn(target_weight.size(1)))
        for _ in range(samples)
    ]
    return qnn_arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: list[int],
    weights: list[torch.Tensor],
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[list[torch.Tensor]]:
    """
    Execute a forward pass through the randomly generated network, storing
    the activation at each layer.  The function returns a list of lists of
    activations for each sample.
    """
    activations = []
    for features, _ in samples:
        layer_outs = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outs.append(current)
        activations.append(layer_outs)
    return activations


def fidelity_adjacency(
    states: list[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Construct a weighted adjacency graph from pairwise state fidelities.
    Edges with fidelity ≥ threshold receive weight 1.0; a secondary
    threshold can add weaker connections.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
        fid = (s1 / (torch.norm(s1) + 1e-12)).dot(
            s2 / (torch.norm(s2) + 1e-12)
        ).item() ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Compact fully‑connected classifier – adapted from Quantum‑NAT seed
# --------------------------------------------------------------------------- #
class QFCModel(nn.Module):
    """
    A lightweight CNN + fully‑connected head that maps a single‑channel
    image to a 4‑dimensional feature vector.  The architecture mirrors
    the classical part of the Quantum‑NAT example and is fully
    differentiable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


# --------------------------------------------------------------------------- #
#  Hybrid model that combines the filter and classifier
# --------------------------------------------------------------------------- #
class HybridConvModel(nn.Module):
    """
    End‑to‑end model that first applies `HybridConvFilter`, optionally
    enriched by a quantum sub‑module, and then passes the result to
    `QFCModel` for classification.  The design allows easy experimentation
    with different quantum back‑ends while keeping the classical surface
    untouched.
    """

    def __init__(self, kernel_size: int = 3, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.filter = HybridConvFilter(kernel_size=kernel_size, conv_threshold=conv_threshold)
        self.classifier = QFCModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        return self.classifier(features)


__all__ = [
    "HybridConvFilter",
    "HybridConvModel",
    "QFCModel",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
]
