"""Hybrid classical GraphQNN incorporating self‑attention, estimation, and quanvolution.

The module mirrors the QML API but is fully classical, using PyTorch, NumPy and networkx.
It exposes a single ``HybridGraphQNN`` class that can be instantiated with any combination
of the three sub‑components: self‑attention on node features, a lightweight estimator head,
or a quanvolution filter for image‑like inputs.  The design is intentionally modular
to allow easy ablation studies or joint training with the quantum counterpart.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Basic utilities – identical to the original GraphQNN seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Sub‑module 1 – classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Drop‑in replacement for the quantum self‑attention block."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
# Sub‑module 2 – estimator head
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor – identical to the ML seed."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(inputs)

# --------------------------------------------------------------------------- #
# Sub‑module 3 – quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quanvolution filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear  = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits   = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# Main hybrid class – combines all sub‑modules
# --------------------------------------------------------------------------- #
class HybridGraphQNN(nn.Module):
    """
    A unified, modular GraphQNN that can operate in three distinct modes:

    * **Graph mode** – node features are transformed with optional self‑attention,
      then propagated through a classical feed‑forward network and finally
      evaluated by an estimator head.
    * **Quanvolution mode** – 2‑D images are processed by the quanvolution filter
      followed by a linear classifier.
    * **Hybrid mode** – both graph and quanvolution branches can be active
      (e.g. for multimodal inputs).

    Parameters
    ----------
    graph_arch : Sequence[int]
        Layer widths for the underlying feed‑forward network.
    use_self_attention : bool, default=True
        Whether to apply self‑attention on node features.
    use_estimator : bool, default=True
        Whether to append the lightweight estimator head.
    use_quanvolution : bool, default=False
        Whether to include the quanvolution pipeline.
    """
    def __init__(
        self,
        graph_arch: Sequence[int],
        use_self_attention: bool = True,
        use_estimator: bool = True,
        use_quanvolution: bool = False,
    ) -> None:
        super().__init__()
        self.graph_arch = list(graph_arch)
        self.use_self_attention = use_self_attention
        self.use_estimator      = use_estimator
        self.use_quanvolution   = use_quanvolution

        # Build sub‑modules
        if use_self_attention:
            self.attention = ClassicalSelfAttention(embed_dim=4)
        if use_estimator:
            self.estimator = EstimatorNN()
        if use_quanvolution:
            self.qfilter   = QuanvolutionFilter()
            self.classifier = nn.Linear(4 * 14 * 14, 10)

        # Random classical graph network – used only for graph mode
        _, self.weights, _, _ = random_network(self.graph_arch, samples=100)

    # --------------------------------------------------------------------- #
    # Helper for graph mode
    # --------------------------------------------------------------------- #
    def _graph_forward(self, inputs: Tensor, graph: nx.Graph | None = None) -> Tensor:
        """
        Forward pass in graph mode.

        Parameters
        ----------
        inputs : Tensor
            Node feature matrix of shape ``(N, F)``.
        graph : nx.Graph | None
            Optional explicit graph; if omitted a simple chain is used.
        """
        if graph is None:
            graph = nx.Graph()
            graph.add_nodes_from(range(inputs.shape[0]))

        # Self‑attention message
        if self.use_self_attention:
            # Dummy parameters – in a real model these would be learnable
            rot = np.zeros((inputs.shape[1],))
            ent = np.zeros((inputs.shape[1],))
            attn_out = self.attention.run(rot, ent, inputs.numpy())
            embeddings = torch.as_tensor(attn_out, dtype=torch.float32)
        else:
            embeddings = inputs

        # Build adjacency from fidelities
        adjacency = fidelity_adjacency(embeddings, threshold=0.5)

        # Feed‑forward propagation over the network
        samples = [(embeddings[i], None) for i in adjacency.nodes()]
        activations = feedforward(self.graph_arch, self.weights, samples)
        final = activations[-1][-1]  # last layer of the last sample

        if self.use_estimator:
            return self.estimator(final)
        return final

    # --------------------------------------------------------------------- #
    # Helper for quanvolution mode
    # --------------------------------------------------------------------- #
    def _quanvolution_forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass in quanvolution mode.

        Parameters
        ----------
        inputs : Tensor
            Image tensor of shape ``(B, 1, 28, 28)``.
        """
        features = self.qfilter(inputs)
        logits   = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    # --------------------------------------------------------------------- #
    # Public forward
    # --------------------------------------------------------------------- #
    def forward(self, inputs: Tensor, graph: nx.Graph | None = None) -> Tensor:
        """
        Dispatches the input through the appropriate sub‑module.

        Parameters
        ----------
        inputs : Tensor
            Either a node feature matrix ``(N, F)`` or an image batch ``(B,1,28,28)``.
        graph : nx.Graph | None
            Optional graph when operating in graph mode.
        """
        if self.use_quanvolution and inputs.ndim == 4 and inputs.shape[2:] == (28, 28):
            return self._quanvolution_forward(inputs)
        return self._graph_forward(inputs, graph)

__all__ = [
    "HybridGraphQNN",
    "ClassicalSelfAttention",
    "EstimatorNN",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
