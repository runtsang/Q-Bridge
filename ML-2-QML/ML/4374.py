"""Enhanced quanvolution classifier – classical implementation.

This module fuses ideas from the original quanvolution filter, a lightweight
FastBaseEstimator for rapid evaluation, and optional LSTM and graph‑based
features inspired by GraphQNN.  The class is fully compatible with the
anchor path ``Quanvolution.py`` but extends it with configurable
quantum‑style behaviour (purely classical here).

Key design points
-----------------
* ``use_lstm`` – when True, a classical LSTM processes the flattened
  convolutional features before the final linear head.
* ``use_graph`` – when True, a simple graph adjacency is built from the
  batch of feature vectors using a distance threshold; the resulting
  adjacency matrix is concatenated to the features.
* ``FastEstimator`` – a thin wrapper around the original FastEstimator
  that accepts the same ``observables`` and ``parameter_sets`` API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Graph utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #

def _euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b).item())


def build_graph(features: torch.Tensor, threshold: float = 0.7) -> nx.Graph:
    """Construct a simple undirected graph where edges connect
    feature vectors whose Euclidean distance is below *threshold*.
    """
    graph = nx.Graph()
    n = features.size(0)
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = _euclidean_distance(features[i], features[j])
            if dist <= threshold:
                graph.add_edge(i, j, weight=1.0)
    return graph


def graph_features(features: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """Return a flattened adjacency matrix that can be concatenated to
    the feature vector.  The adjacency is symmetric and zero‑diagonal.
    """
    graph = build_graph(features, threshold)
    adj = nx.to_numpy_array(graph, dtype=np.float32)
    return torch.from_numpy(adj).to(features.device).reshape(features.size(0), -1)


# --------------------------------------------------------------------------- #
#  Classical quanvolution components
# --------------------------------------------------------------------------- #

class ClassicalQuanvolutionFilter(nn.Module):
    """A lightweight 2‑D convolution that mimics the patch‑wise approach
    of the original quanvolution filter but stays fully classical.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per sample


# --------------------------------------------------------------------------- #
#  Main classifier
# --------------------------------------------------------------------------- #

class EnhancedQuanvolutionClassifier(nn.Module):
    """
    Classical quanvolution classifier with optional LSTM and graph features.

    Parameters
    ----------
    use_lstm : bool, default False
        If True, a bidirectional LSTM processes the flattened features
        before the final linear head.
    use_graph : bool, default False
        If True, a simple adjacency matrix is computed from the feature
        vectors and concatenated to the input of the final linear layer.
    """
    def __init__(self, use_lstm: bool = False, use_graph: bool = False) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.use_graph = use_graph

        self.qfilter = ClassicalQuanvolutionFilter()
        feature_dim = 4 * 14 * 14  # matches original architecture

        if self.use_lstm:
            self.lstm = nn.LSTM(feature_dim, feature_dim // 2, batch_first=True, bidirectional=True)
            lstm_out_dim = feature_dim
        else:
            lstm_out_dim = feature_dim

        if self.use_graph:
            self.graph_linear = nn.Linear(lstm_out_dim * 2, lstm_out_dim)

        self.classifier = nn.Linear(lstm_out_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.qfilter(x)  # (batch, feature_dim)

        if self.use_lstm:
            # LSTM expects (batch, seq_len, input_size); use seq_len=1
            lstm_out, _ = self.lstm(features.unsqueeze(1))
            features = lstm_out.squeeze(1)

        if self.use_graph:
            graph_feat = graph_features(features)
            features = torch.cat([features, graph_feat], dim=1)
            features = self.graph_linear(features)

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["EnhancedQuanvolutionClassifier", "FastEstimator", "FastBaseEstimator"]
